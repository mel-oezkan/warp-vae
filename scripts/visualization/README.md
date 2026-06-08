# Visualization Scripts

Standalone plotting/visualization utilities for this project. Each script is
runnable from the repo root and inserts the project root on `sys.path`
automatically, so the usual pattern is:

```bash
python scripts/visualization/<script>.py [args...]
```

Most scripts accept `--help` for the full argument list. Run them from the
project root with `conda activate cv`.

---

## Camera / dataset geometry

| Script | What it does |
|--------|--------------|
| [plot_camera_positions.py](plot_camera_positions.py) | One scatter plot per CO3D sequence with shared axis scales, so camera trajectories can be visually compared across sequences. |
| [plot_camera_distance_to_center.py](plot_camera_distance_to_center.py) | Per-frame camera distance to the camera-centroid for many CO3D sequences in a single figure. Highlights highest/lowest/median sequences as outliers. |
| [plot_sequence_frames.py](plot_sequence_frames.py) | Strip of sampled RGB frames per CO3D sequence (`--step` to control frame spacing). |
| [plot_mvi2_cameras.py](plot_mvi2_cameras.py) | Reads COLMAP poses (`sparse/0/*.bin`) from MVImgNet2 sequences via pycolmap, recovers camera centers/directions, and scatters the capture trajectories (one color per sequence). |
| [plot_mvi2_overlap.py](plot_mvi2_overlap.py) | For two MVImgNet2 frames, splits observed COLMAP 3D points into only-A / only-B / shared, overlays them on the pixels, draws shared correspondences, and reports the overlap (Jaccard IoU). |
| [plot_omni_camera_distribution.py](plot_omni_camera_distribution.py) | OmniObject3D camera distribution: converts (elevation, azimuth) to unit-sphere positions and shows equirectangular + top-down views shared across all instances. |

## RoMA correspondences & warps

| Script | What it does |
|--------|--------------|
| [visualize_roma_matches.py](visualize_roma_matches.py) | Side-by-side image pair with RoMA match lines colored by confidence. Works on explicit `--img_a/--img_b` or an auto-picked CO3D pair. |
| [visualize_roma_warps_grid.py](visualize_roma_warps_grid.py) | 2×2 grid: original A, original B, B→A warp, A→B warp. White = low-confidence/out-of-view. |
| [visualize_roma_warps.py](visualize_roma_warps.py) | Fuller RoMA visualization — correspondences, confidence maps, and VAE latent embeddings for CO3D + OmniObject3D pairs. Writes to `eval_outputs/roma_visualization/`. |

## Depth-based warps

| Script | What it does |
|--------|--------------|
| [visualize_depth_warps.py](visualize_depth_warps.py) | Depth-based warp generation on CO3D: images, depth maps, A→B confidence, warped B, and a checkerboard overlay of A vs warped B. |

## Distance-sweep diagnostics

These re-plot or render the outputs of a distance sweep stored under
`<sweep_dir>/per_pair.csv`. The `da3_*` variants work with DepthAnything3 (DA3)
depth-based warps; the `roma_*` variants with RoMA2 warps.

| Script | What it does |
|--------|--------------|
| [plot_da3_sweep.py](plot_da3_sweep.py) | Re-plots a DA3 distance sweep from `per_pair.csv` (no recomputation). Highlights 3 best / worst / closest-to-mean sequences by mean warp L1. |
| [plot_da3_sweep_pairs.py](plot_da3_sweep_pairs.py) | Per-sequence 6-column pair grids (imA, imB, B→A, A→B, DA3 depth A, DA3 depth B). Recomputes warps + depth on the fly. |
| [plot_da3_sweep_good_bad.py](plot_da3_sweep_good_bad.py) | Per-(bin, sequence) grids contrasting high-confidence vs low-confidence pairs. Output under `<sweep_dir>/good_bad_grids/`. |
| [plot_da3_sweep_conf_hist.py](plot_da3_sweep_conf_hist.py) | Small-multiple histograms of DA3 warp confidence, one per distance bin, with mean/median overlays. |
| [plot_roma_sweep.py](plot_roma_sweep.py) | RoMA counterpart of `plot_da3_sweep.py`, with auto-detection of RoMA metric columns across CSV schemas. |
| [plot_roma_sweep_pairs.py](plot_roma_sweep_pairs.py) | RoMA counterpart of `plot_da3_sweep_pairs.py` — pair grids with warp-field maps; low-confidence regions filled with a translucent target background. |
| [plot_roma_sweep_conf_hist.py](plot_roma_sweep_conf_hist.py) | RoMA counterpart of `plot_da3_sweep_conf_hist.py` — per-bin confidence histograms. |
| [visualize_distance_sampling.py](visualize_distance_sampling.py) | Samples pairs by camera distance, computes RoMA warps, warps in pixel space, and saves source/target/warped images with L2 error heatmaps. |

## VAE latents & reconstructions

| Script | What it does |
|--------|--------------|
| [visualize_latent_pca_grid.py](visualize_latent_pca_grid.py) | PCA-RGB visualization of VAE latents across multiple checkpoints/models, objects, and views in one grid (CO3D). |
| [visualize_latent_warp_grid.py](visualize_latent_warp_grid.py) | Latent-space warping with an EQ-VAE model for a pair: originals, reconstructions, and warped-latent decodes under three background fill modes (white/source/target). |
| [visualize_warp_dataset.py](visualize_warp_dataset.py) | Inspects `WarpCO3DDataset` outputs: source/target images, A→B / B→A warp maps, confidences, warped images, and warp errors. |
| [plot_warp_reconstructions.py](plot_warp_reconstructions.py) | Reconstruction comparison grid (rows = objects, columns = original + each model's reconstruction) to surface artifacts/patterns. |
| [plot_omniobject_reconstructions.py](plot_omniobject_reconstructions.py) | OmniObject3D reconstructions: a single-object multi-view figure and a multi-object single-view figure (source vs reconstruction). |
