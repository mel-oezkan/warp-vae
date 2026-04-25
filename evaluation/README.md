# EQVAE Evaluation System

Comprehensive validation toolkit for trained EQVAE models.

## Quick Start

### Automated Evaluation

Run full evaluation pipeline:

```bash
python evaluation/evaluate_eqvae.py \
    --checkpoint "checkpoints/loose-mushroom-of-algebraic-tempering_EQ-VAE small model for GPU memory testing/last.ckpt" \
    --config config/eqvae_omniobject_small.yaml \
    --output_dir evaluation_outputs/eqvae_small_epoch009 \
    --use_ema \
    --batch_size 8
```

### Interactive Exploration

Use Jupyter notebook:

```bash
cd evaluation/notebooks
jupyter notebook interactive_validation.ipynb
```

## What Gets Evaluated

### Reconstruction Quality
- **MSE** - Mean Squared Error
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **LPIPS** - Learned Perceptual Image Patch Similarity
- **FID** - Fréchet Inception Distance

### Equivariance Properties
- **Scale Equivariance** - Tests transformation consistency for scaling (0.25x, 0.5x, 0.75x, 1.0x)
- **Rotation Equivariance** - Tests transformation consistency for rotations (0°, 90°, 180°, 270°)
- **Combined Metrics** - Mean, std, and max errors across all transformations

### Latent Space Analysis
- **t-SNE Visualization** - 2D projection of latent space
- **Distribution Analysis** - Per-channel statistics and KL divergence from N(0,1)
- **Interpolations** - Linear interpolation between latent codes

### Multi-View Consistency
- **Latent Similarity** - Cosine similarity across views
- **PCA Trajectory** - Latent space trajectory visualization
- **Adjacent Distances** - L2 distances between sequential views

## Output Structure

```
evaluation_outputs/
├── metrics/
│   └── eqvae_epoch9_metrics.json      # All computed metrics
├── figures/
│   ├── reconstruction_grid.pdf         # Original vs reconstruction grid
│   ├── equivariance_tests.pdf          # Transformation test matrix
│   ├── latent_tsne.pdf                 # t-SNE projection
│   ├── latent_distributions.pdf        # Latent statistics
│   ├── multiview_consistency.pdf       # Multi-view analysis
│   └── interpolations.pdf              # Latent interpolations
└── reports/
    └── validation_report.md            # Auto-generated summary
```

## Dependencies

Install required packages:

```bash
pip install pytorch-fid scikit-learn scikit-image matplotlib seaborn tqdm
```

## Module Structure

```
evaluation/
├── evaluator.py                # Main evaluator class
├── evaluate_eqvae.py          # CLI script
├── metrics/                    # Metric calculators
│   ├── reconstruction_metrics.py
│   ├── lpips_metric.py
│   ├── equivariance_metrics.py
│   └── fid_score.py
├── visualizers/               # Visualization generators
│   ├── reconstruction_viz.py
│   ├── latent_viz.py
│   ├── equivariance_viz.py
│   └── multiview_viz.py
└── notebooks/
    └── interactive_validation.ipynb
```

## Usage Examples

### Custom Batch Size

```bash
python evaluation/evaluate_eqvae.py \
    --checkpoint path/to/checkpoint.ckpt \
    --config path/to/config.yaml \
    --output_dir evaluation_outputs/custom_run \
    --batch_size 4 \
    --num_workers 2
```

### Quick Test (Small Sample)

```bash
python evaluation/evaluate_eqvae.py \
    --checkpoint path/to/checkpoint.ckpt \
    --config path/to/config.yaml \
    --output_dir evaluation_outputs/test \
    --num_fid_samples 500  # Use fewer samples for speed
```

## Programmatic Usage

```python
from evaluation.evaluator import EQVAEEvaluator

# Create evaluator
evaluator = EQVAEEvaluator(
    checkpoint_path="checkpoints/.../last.ckpt",
    config_path="config/eqvae_omniobject_small.yaml",
    output_dir="evaluation_outputs/my_run",
    use_ema=True,
    batch_size=8
)

# Load model
evaluator.load_checkpoint()
evaluator.setup_dataloader()

# Compute specific metrics
metrics = evaluator.compute_all_metrics()

# Generate visualizations
evaluator.generate_all_visualizations(metrics)

# Save results
evaluator.save_metrics(metrics)
evaluator.generate_report(metrics)
```

## Notes

- All evaluation runs in `torch.no_grad()` mode
- EMA weights are used by default if available
- Validation split matches training configuration
- All figures saved in both PDF (publication) and PNG (web) formats
- Metrics saved to JSON for easy parsing and analysis
- GPU memory optimized for GTX 1080 Ti (11GB)

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch_size 4
```

### FID Computation Fails

Make sure pytorch-fid is installed:
```bash
pip install pytorch-fid
```

### Slow Execution

Reduce number of samples:
```bash
--num_fid_samples 1000
```

## Citation

If you use this evaluation system, please cite:

```bibtex
@software{eqvae_evaluation,
  title={EQVAE Evaluation Toolkit},
  year={2026},
  author={Your Name}
}
```
