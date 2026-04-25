"""
Generic VAE evaluator supporting all model variants.

Computes g-FID, s-FID, LPIPS, MSE, PSNR, SSIM for one or more models,
and produces a comparison table + per-model JSON results.
"""

import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from evaluation.model_wrapper import VAEModelWrapper, _resolve_hydra_refs
from src.data.datamodule import VAEDataModule


class VAEEvaluator:
    """
    Model-agnostic evaluator for all VAE variants.

    Supports comparing multiple models on the same dataset.
    """

    def __init__(
        self,
        output_dir: str,
        device: str = 'cuda',
        batch_size: int = 8,
        num_workers: int = 4,
        num_fid_samples: int = 5000,
        num_lpips_samples: Optional[int] = None,
        num_recon_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_fid_samples = num_fid_samples
        self.num_lpips_samples = num_lpips_samples
        self.num_recon_samples = num_recon_samples
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def setup_dataloader(self, config_path: str):
        """Setup validation dataloader from a config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = _resolve_hydra_refs(config)

        data_config = config.get('data', {})

        # Handle both our custom config format and the old LDM format
        if 'params' in data_config and 'dataset_config' in data_config.get('params', {}):
            datamodule = VAEDataModule(
                dataset_config=data_config['params']['dataset_config'],
                batch_size=self.batch_size,
                val_split=data_config['params'].get('val_split', 0.1),
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                seed=self.seed,
            )
            datamodule.setup('fit')
            return datamodule.val_dataloader()
        else:
            raise ValueError(
                f"Unsupported data config format in {config_path}. "
                "Expected data.params.dataset_config."
            )

    def evaluate_model(
        self,
        wrapper: VAEModelWrapper,
        val_loader,
        model_name: str,
        metrics_to_compute: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single model.

        Args:
            wrapper: VAEModelWrapper instance
            val_loader: Validation dataloader
            model_name: Display name for this model
            metrics_to_compute: List of metrics to compute.
                Options: 'reconstruction', 'lpips', 'fid'. Default: all.

        Returns:
            Dictionary with all computed metrics
        """
        if metrics_to_compute is None:
            metrics_to_compute = ['reconstruction', 'lpips', 'fid']

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name} ({wrapper.name})")
        print(f"{'='*60}")

        reconstruct_fn = lambda model, imgs: wrapper.reconstruct(imgs)
        results = {
            'model_name': model_name,
            'model_class': wrapper.name,
            'model_type': wrapper.model_type,
            'timestamp': datetime.now().isoformat(),
        }

        if 'reconstruction' in metrics_to_compute:
            print("\n[1/3] Reconstruction metrics (MSE, PSNR, SSIM)...")
            from evaluation.metrics import ReconstructionMetrics
            recon_calc = ReconstructionMetrics(
                device=self.device, reconstruct_fn=reconstruct_fn,
            )
            results['reconstruction'] = recon_calc.compute(
                val_loader, wrapper.model, num_samples=self.num_recon_samples,
            )

        if 'lpips' in metrics_to_compute:
            print("\n[2/3] LPIPS...")
            from evaluation.metrics import LPIPSCalculator
            lpips_calc = LPIPSCalculator(
                device=self.device, reconstruct_fn=reconstruct_fn,
            )
            results['lpips'] = lpips_calc.compute(
                val_loader, wrapper.model, num_samples=self.num_lpips_samples,
            )

        if 'fid' in metrics_to_compute:
            print("\n[3/3] g-FID & s-FID...")
            from evaluation.metrics import FIDCalculator
            fid_calc = FIDCalculator(
                device=self.device, reconstruct_fn=reconstruct_fn,
            )
            results['fid'] = fid_calc.compute(
                val_loader, wrapper.model, num_samples=self.num_fid_samples,
            )

        return results

    def compare_models(
        self,
        checkpoints: List[str],
        configs: List[str],
        model_names: Optional[List[str]] = None,
        data_config: Optional[str] = None,
        metrics_to_compute: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate and compare multiple models.

        Args:
            checkpoints: List of checkpoint paths
            configs: List of config paths (one per checkpoint, or one shared)
            model_names: Display names (defaults to checkpoint filenames)
            data_config: Config to use for dataloader setup.
                        Defaults to first config in configs list.
            metrics_to_compute: Which metrics to compute

        Returns:
            Combined results dict with per-model metrics and comparison table
        """
        assert len(checkpoints) > 0, "Need at least one checkpoint"
        if len(configs) == 1 and len(checkpoints) > 1:
            configs = configs * len(checkpoints)
        assert len(configs) == len(checkpoints), \
            f"Need one config per checkpoint (got {len(configs)} configs, {len(checkpoints)} checkpoints)"

        if model_names is None:
            model_names = [Path(c).parent.name for c in checkpoints]

        # Setup dataloader (use data_config or first config)
        loader_config = data_config or configs[0]
        print(f"Setting up dataloader from: {Path(loader_config).name}")
        val_loader = self.setup_dataloader(loader_config)

        all_results = {}
        for ckpt, cfg, name in zip(checkpoints, configs, model_names):
            wrapper = VAEModelWrapper.from_config(cfg, ckpt, self.device)
            results = self.evaluate_model(wrapper, val_loader, name, metrics_to_compute)
            all_results[name] = results

            # Save per-model results
            safe_name = name.replace(" ", "_").replace("/", "_")
            model_path = self.output_dir / f"{safe_name}_metrics.json"
            with open(model_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Free GPU memory
            del wrapper
            torch.cuda.empty_cache()

        # Build and save comparison
        comparison = self._build_comparison_table(all_results)
        all_results['_comparison'] = comparison

        # Save combined results
        combined_path = self.output_dir / "comparison.json"
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print comparison table
        self._print_comparison(comparison)

        return all_results

    def _build_comparison_table(self, all_results: Dict) -> Dict:
        """Build a flat comparison table from per-model results."""
        rows = []
        for name, res in all_results.items():
            row = {'model': name}
            if 'reconstruction' in res:
                row.update({
                    'MSE': res['reconstruction']['mse'],
                    'PSNR': res['reconstruction']['psnr'],
                    'SSIM': res['reconstruction']['ssim'],
                })
            if 'lpips' in res:
                row['LPIPS'] = res['lpips']['mean']
            if 'fid' in res:
                row['g-FID'] = res['fid']['g_fid']
                row['s-FID'] = res['fid']['s_fid']
            rows.append(row)
        return rows

    def _print_comparison(self, comparison: List[Dict]):
        """Pretty-print comparison table."""
        if not comparison:
            return

        print(f"\n{'='*80}")
        print("  COMPARISON TABLE")
        print(f"{'='*80}")

        # Collect all metric keys
        metric_keys = [k for k in comparison[0].keys() if k != 'model']
        header = f"{'Model':<25}"
        for k in metric_keys:
            header += f" {k:>10}"
        print(header)
        print("-" * len(header))

        for row in comparison:
            line = f"{row['model']:<25}"
            for k in metric_keys:
                val = row.get(k)
                if val is None:
                    line += f" {'N/A':>10}"
                elif isinstance(val, float):
                    line += f" {val:>10.4f}"
                else:
                    line += f" {str(val):>10}"
            print(line)

        print(f"{'='*80}")
        print(f"Results saved to: {self.output_dir}")
