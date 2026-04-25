#!/usr/bin/env python
"""
Evaluate and compare VAE models: g-FID, s-FID, LPIPS, MSE, PSNR, SSIM.

Supports all model variants (Vanilla, EQ-VAE, Warp, Plucker, etc.).

Single model:
    python evaluation/evaluate_eqvae.py \
        --checkpoints checkpoints/my_model/last.ckpt \
        --configs config/vanilla_vae_co3d.yaml \
        --model_names "Vanilla VAE" \
        --output_dir evaluation_outputs/vanilla

Compare multiple models:
    python evaluation/evaluate_eqvae.py \
        --checkpoints \
            checkpoints/vanilla/last.ckpt \
            checkpoints/warp/last.ckpt \
            checkpoints/eqvae/last.ckpt \
        --configs \
            config/vanilla_vae_co3d.yaml \
            config/warp_vae_co3d_precomputed.yaml \
            config/eqvae_co3d_hydrant_50seq.yaml \
        --model_names "Vanilla" "Warp-VAE" "EQ-VAE" \
        --output_dir evaluation_outputs/comparison

Shared data config (all models evaluated on same dataset):
    python evaluation/evaluate_eqvae.py \
        --checkpoints ckpt1.ckpt ckpt2.ckpt \
        --configs config1.yaml config2.yaml \
        --data_config config/vanilla_vae_co3d.yaml \
        --output_dir evaluation_outputs/comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import VAEEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VAE models: g-FID, s-FID, LPIPS, reconstruction metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--checkpoints', nargs='+', required=True,
        help='Checkpoint path(s)',
    )
    parser.add_argument(
        '--configs', nargs='+', required=True,
        help='Config path(s). One per checkpoint, or one shared config.',
    )
    parser.add_argument(
        '--model_names', nargs='+', default=None,
        help='Display names for models (defaults to checkpoint folder names)',
    )
    parser.add_argument(
        '--data_config', type=str, default=None,
        help='Config to use for dataloader (defaults to first --configs entry)',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save results',
    )
    parser.add_argument(
        '--metrics', nargs='+',
        default=['reconstruction', 'lpips', 'fid'],
        choices=['reconstruction', 'lpips', 'fid'],
        help='Which metrics to compute',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_fid_samples', type=int, default=5000)
    parser.add_argument('--num_lpips_samples', type=int, default=None)
    parser.add_argument('--num_recon_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Validate checkpoint paths
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"Error: Checkpoint not found: {ckpt}")
            sys.exit(1)
    for cfg in args.configs:
        if not Path(cfg).exists():
            print(f"Error: Config not found: {cfg}")
            sys.exit(1)

    evaluator = VAEEvaluator(
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_fid_samples=args.num_fid_samples,
        num_lpips_samples=args.num_lpips_samples,
        num_recon_samples=args.num_recon_samples,
        seed=args.seed,
    )

    try:
        evaluator.compare_models(
            checkpoints=args.checkpoints,
            configs=args.configs,
            model_names=args.model_names,
            data_config=args.data_config,
            metrics_to_compute=args.metrics,
        )
        print("\nEvaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
