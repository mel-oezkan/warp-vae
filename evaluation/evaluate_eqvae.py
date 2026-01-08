#!/usr/bin/env python
"""
Automated EQVAE validation script.

Usage:
    python evaluation/evaluate_eqvae.py \
        --checkpoint checkpoints/.../last.ckpt \
        --config config/eqvae_omniobject_small.yaml \
        --output_dir evaluation_outputs/eqvae_small_epoch009 \
        --use_ema \
        --num_fid_samples 5000 \
        --batch_size 8
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import EQVAEEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive EQVAE model validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save evaluation outputs'
    )

    # Optional arguments
    parser.add_argument(
        '--use_ema',
        action='store_true',
        help='Use EMA weights if available'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--num_fid_samples',
        type=int,
        default=5000,
        help='Number of samples for FID computation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    # Create evaluator
    evaluator = EQVAEEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        use_ema=args.use_ema,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Run full evaluation
    try:
        evaluator.run_full_evaluation()
        print("\n✅ Evaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
