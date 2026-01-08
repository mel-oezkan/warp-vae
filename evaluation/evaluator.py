"""
Main evaluator class for EQVAE model validation.
"""

import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm

from ldm.util import instantiate_from_config
from src.data.datamodule import VAEDataModule


class EQVAEEvaluator:
    """
    Comprehensive evaluator for trained EQVAE models.

    Handles checkpoint loading, metric computation, visualization generation,
    and report creation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        output_dir: str,
        device: str = 'cuda',
        use_ema: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize EQVAE evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config YAML
            output_dir: Directory to save outputs
            device: Device to run evaluation on
            use_ema: Whether to use EMA weights
            batch_size: Batch size for evaluation
            num_workers: Number of dataloader workers
            seed: Random seed for reproducibility
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.use_ema = use_ema
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Create output directories
        self.metrics_dir = self.output_dir / "metrics"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.metrics_dir, self.figures_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize components
        self.model = None
        self.datamodule = None
        self.config = None

        print("[EQVAEEvaluator] Initialized")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Config: {self.config_path}")
        print(f"  Output: {self.output_dir}")
        print(f"  Device: {self.device}")
        print(f"  Use EMA: {self.use_ema}")

    def load_checkpoint(self):
        """Load EQVAE model from checkpoint."""
        print("\n" + "="*60)
        print("Loading Checkpoint")
        print("="*60)

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print("✓ Config loaded")

        # Instantiate model from config
        model = instantiate_from_config(self.config['model'])
        print(f"✓ Model instantiated: {self.config['model']['target']}")

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')

        # Extract state dict (handle different checkpoint formats)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Remove 'model.' prefix if present
        model_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"  Warning: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {len(unexpected_keys)}")

        print(f"✓ Checkpoint loaded")

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        # Use EMA weights if available
        if self.use_ema and hasattr(model, 'model_ema') and model.model_ema is not None:
            print(f"✓ Using EMA weights")
            self.model = model
            self._ema_mode = True
        else:
            self.model = model
            self._ema_mode = False

        print(f"✓ Model ready on {self.device}")

        # Extract epoch from checkpoint if available
        self.epoch = ckpt.get('epoch', 'unknown')
        print(f"✓ Epoch: {self.epoch}")

        return self.model

    def setup_dataloader(self, split='val', sample_mode='single'):
        """
        Setup dataloader for evaluation.

        Args:
            split: Dataset split ('val' or 'test')
            sample_mode: Sampling mode ('single' or 'pairs')
        """
        print("\n" + "="*60)
        print("Setting up Dataloader")
        print("="*60)

        # Get dataset config from model config
        data_config = self.config.get('data', {})

        # Create datamodule
        self.datamodule = VAEDataModule(
            dataset_config=data_config['params']['dataset_config'],
            batch_size=self.batch_size,
            val_split=data_config['params'].get('val_split', 0.1),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            seed=self.seed,
        )

        # Setup
        self.datamodule.setup('fit')

        # Get dataloader
        if split == 'val':
            self.val_loader = self.datamodule.val_dataloader()
            print(f"✓ Validation dataloader ready")
            print(f"  Validation samples: {len(self.datamodule.val_dataset)}")
            print(f"  Batches: {len(self.val_loader)}")
            print(f"  Batch size: {self.batch_size}")
            return self.val_loader
        else:
            raise NotImplementedError(f"Split '{split}' not implemented")

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics.

        Returns:
            Dictionary with all computed metrics
        """
        print("\n" + "="*60)
        print("Computing Metrics")
        print("="*60)

        metrics = {
            'checkpoint': str(self.checkpoint_path),
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'use_ema': self.use_ema,
        }

        # Import metrics modules
        from evaluation.metrics import (
            ReconstructionMetrics,
            LPIPSCalculator,
            EquivarianceMetrics,
            FIDCalculator,
        )

        # Compute reconstruction metrics
        print("\n1. Reconstruction Metrics...")
        recon_metrics = ReconstructionMetrics(self.model, self.device)
        metrics['reconstruction_quality'] = recon_metrics.compute(self.val_loader)

        # Compute LPIPS
        print("\n2. LPIPS Perceptual Similarity...")
        lpips_calc = LPIPSCalculator(self.model)
        metrics['lpips'] = lpips_calc.compute(self.val_loader)

        # Compute equivariance metrics
        print("\n3. Equivariance Properties...")
        equiv_metrics = EquivarianceMetrics(self.model, self.device)
        metrics['equivariance'] = equiv_metrics.compute(self.val_loader, num_samples=500)

        # Compute FID (optional, can be slow)
        print("\n4. FID Score...")
        try:
            fid_calc = FIDCalculator(device=self.device)
            metrics['fid'] = fid_calc.compute_fid(self.val_loader, self.model, num_samples=5000)
        except Exception as e:
            print(f"  Warning: FID computation failed: {e}")
            metrics['fid'] = None

        # Compute latent space statistics
        print("\n5. Latent Space Statistics...")
        metrics['latent_space'] = self._compute_latent_stats()

        print("\n✓ All metrics computed")
        return metrics

    def _compute_latent_stats(self) -> Dict[str, Any]:
        """Compute latent space statistics."""
        latents = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Extracting latents")):
                if batch_idx >= 200:  # Limit to 200 batches for speed
                    break

                images = batch['image'].to(self.device)

                # Encode
                if self._ema_mode and hasattr(self.model, 'ema_scope'):
                    with self.model.ema_scope():
                        posterior = self.model.encode(images)
                else:
                    posterior = self.model.encode(images)

                z = posterior.sample()

                # Flatten spatial dimensions and collect
                z_flat = z.view(z.size(0), z.size(1), -1).mean(dim=2)  # [B, C]
                latents.append(z_flat.cpu())

        latents = torch.cat(latents, dim=0)  # [N, C]

        # Compute statistics
        stats = {
            'mean': latents.mean(dim=0).tolist(),
            'std': latents.std(dim=0).tolist(),
            'min': latents.min(dim=0).values.tolist(),
            'max': latents.max(dim=0).values.tolist(),
        }

        # KL divergence from N(0,1)
        kl_div = 0.5 * torch.sum(
            latents.mean(dim=0)**2 + latents.std(dim=0)**2 - 1 - torch.log(latents.std(dim=0)**2)
        ).item()
        stats['kl_divergence'] = kl_div

        # Sparsity
        sparsity = (torch.abs(latents) < 0.1).float().mean().item()
        stats['sparsity_ratio'] = sparsity

        return stats

    def generate_all_visualizations(self, metrics: Dict[str, Any]):
        """
        Generate all visualizations.

        Args:
            metrics: Computed metrics dictionary
        """
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)

        from evaluation.visualizers import (
            ReconstructionVisualizer,
            LatentVisualizer,
            EquivarianceVisualizer,
            MultiViewVisualizer,
        )

        # 1. Reconstruction grid
        print("\n1. Reconstruction Grid...")
        recon_viz = ReconstructionVisualizer(self.model, self.device)
        recon_viz.create_reconstruction_grid(
            self.val_loader,
            num_samples=16,
            save_path=self.figures_dir / "reconstruction_grid"
        )

        # 2. Equivariance tests
        print("\n2. Equivariance Tests...")
        equiv_viz = EquivarianceVisualizer(self.model, self.device)
        equiv_viz.visualize_transformation_tests(
            self.val_loader,
            num_samples=6,
            save_path=self.figures_dir / "equivariance_tests"
        )

        # 3. Latent space visualizations
        print("\n3. Latent Space Visualizations...")
        latent_viz = LatentVisualizer(self.model, self.device)

        # t-SNE
        latent_viz.visualize_latent_tsne(
            self.val_loader,
            num_samples=2000,
            save_path=self.figures_dir / "latent_tsne"
        )

        # Distributions
        latent_viz.visualize_latent_distributions(
            self.val_loader,
            num_samples=2000,
            save_path=self.figures_dir / "latent_distributions"
        )

        # Interpolations
        latent_viz.visualize_interpolations(
            self.val_loader,
            num_pairs=4,
            num_steps=10,
            save_path=self.figures_dir / "interpolations"
        )

        # 4. Multi-view consistency
        print("\n4. Multi-View Consistency...")
        try:
            multiview_viz = MultiViewVisualizer(self.model, self.device)
            multiview_viz.visualize_24_view_consistency(
                self.val_loader,
                save_path=self.figures_dir / "multiview_consistency"
            )
        except Exception as e:
            print(f"  Warning: Multi-view visualization failed: {e}")

        print("\n✓ All visualizations generated")

    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSON file."""
        metrics_path = self.metrics_dir / f"eqvae_epoch{self.epoch}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_path}")

    def generate_report(self, metrics: Dict[str, Any]):
        """Generate markdown report."""
        report_path = self.reports_dir / "validation_report.md"

        with open(report_path, 'w') as f:
            f.write(f"# EQVAE Validation Report\n\n")
            f.write(f"**Date:** {metrics['timestamp']}\n\n")
            f.write(f"**Checkpoint:** `{metrics['checkpoint']}`\n\n")
            f.write(f"**Epoch:** {metrics['epoch']}\n\n")
            f.write(f"**Use EMA:** {metrics['use_ema']}\n\n")
            f.write(f"---\n\n")

            # Reconstruction quality
            f.write(f"## Reconstruction Quality\n\n")
            if 'reconstruction_quality' in metrics:
                for k, v in metrics['reconstruction_quality'].items():
                    f.write(f"- **{k.upper()}:** {v:.4f}\n")
            f.write(f"\n")

            # LPIPS
            if 'lpips' in metrics and metrics['lpips']:
                f.write(f"## LPIPS Perceptual Similarity\n\n")
                f.write(f"- **Mean:** {metrics['lpips']['mean']:.4f}\n")
                f.write(f"- **Std:** {metrics['lpips']['std']:.4f}\n")
                f.write(f"\n")

            # FID
            if 'fid' in metrics and metrics['fid'] is not None:
                f.write(f"## FID Score\n\n")
                f.write(f"- **FID:** {metrics['fid']:.2f}\n")
                f.write(f"\n")

            # Equivariance
            if 'equivariance' in metrics:
                f.write(f"## Equivariance Properties\n\n")
                f.write(f"```json\n{json.dumps(metrics['equivariance'], indent=2)}\n```\n\n")

            # Latent space
            if 'latent_space' in metrics:
                f.write(f"## Latent Space Statistics\n\n")
                f.write(f"- **KL Divergence:** {metrics['latent_space']['kl_divergence']:.4f}\n")
                f.write(f"- **Sparsity Ratio:** {metrics['latent_space']['sparsity_ratio']:.4f}\n")
                f.write(f"\n")

            # Figures
            f.write(f"## Figures\n\n")
            f.write(f"- [Reconstruction Grid](../figures/reconstruction_grid.pdf)\n")
            f.write(f"- [Equivariance Tests](../figures/equivariance_tests.pdf)\n")
            f.write(f"- [Latent t-SNE](../figures/latent_tsne.pdf)\n")
            f.write(f"- [Latent Distributions](../figures/latent_distributions.pdf)\n")
            f.write(f"- [Interpolations](../figures/interpolations.pdf)\n")
            f.write(f"- [Multi-View Consistency](../figures/multiview_consistency.pdf)\n")

        print(f"✓ Report saved to: {report_path}")

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*70)
        print(" "*20 + "EQVAE MODEL VALIDATION")
        print("="*70)

        # Load checkpoint
        self.load_checkpoint()

        # Setup dataloader
        self.setup_dataloader()

        # Compute metrics
        metrics = self.compute_all_metrics()

        # Save metrics
        self.save_metrics(metrics)

        # Generate visualizations
        self.generate_all_visualizations(metrics)

        # Generate report
        self.generate_report(metrics)

        print("\n" + "="*70)
        print(" "*25 + "EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Metrics: {self.metrics_dir}")
        print(f"  - Figures: {self.figures_dir}")
        print(f"  - Report: {self.reports_dir}")
