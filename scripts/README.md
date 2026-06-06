# `scripts/` layout and conventions

Scripts are grouped by topic. `scripts/` is on `sys.path` (via the `cv_project.pth`
file in the conda env), so the subfolders are importable as top-level packages —
e.g. `from warps.precompute_depth_warps import load_annotations`.

## Folders

| Folder            | Contents |
|-------------------|----------|
| `preprocessing/`  | Dataset prep and downloads (CO3D / OmniObject3D / MVImgNet, DepthAnything prediction). |
| `warps/`          | Warp precomputation (RoMA / depth / DA3) and hydrant pair screening + sweeps. |
| `sweeps/`         | VAE probing experiments (equivariance, RoMA, salt&pepper) and distance/angle sweeps. |
| `analysis/`       | Multi-view latent consistency and camera analysis, latent comparison. |
| `visualization/`  | `visualize_*` tools and `plot_*` figure generators. |
| `demos/`          | Self-contained demos and quick experiments (`*_demo.py`, latent interp/inpaint). |
| `shell/`          | Shell launchers / job-queue scripts (`.sh`). |
| `data_process/`   | Importable dataset utilities package (`from data_process... import ...`). |

`setup_third_party.sh`, `move_script_artifacts.sh`, and this README stay at the top level.

## Conventions

- Keep source scripts (`.py`, `.sh`) under the topic folders above.
- Do not keep generated artifacts (images, CSVs) here; scripts write them to `outputs/scripts/`.
- Use `./move_script_artifacts.sh` to move stray `.png` / `.csv` files from `scripts/` to `outputs/scripts/`.
- Compiled caches and artifacts are ignored via `.gitignore` in this folder.
