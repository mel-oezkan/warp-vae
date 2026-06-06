#!/usr/bin/env python3
"""Visualize PCA-colored latents in a grid: columns = objects, rows = models.

Usage:
    python scripts/visualization/visualize_latent_pca_grid.py \
        --checkpoints ckpt1.ckpt ckpt2.ckpt weights/f8/model.ckpt \
        --configs cfg1.yaml cfg2.yaml config/baseVAE.yaml \
        --model_names "Warp-VAE (cosine)" "Warp-VAE (toybus-hp)" "SD-VAE" \
        --dataset co3d_native \
        --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
        --num_objects 5 \
        --views_per_object 3 \
        --output latent_pca_grid.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis import load_model, encode_images, denormalize


def main():
    parser = argparse.ArgumentParser(description="PCA latent grid visualization")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--model_types", nargs="+", default=None,
                        help="Model types (auto-detected if not specified)")
    parser.add_argument("--dataset", default="co3d_native", choices=["co3d_native"])
    parser.add_argument("--co3d_native_dir", type=str, required=True)
    parser.add_argument("--num_objects", type=int, default=5,
                        help="Number of objects (columns)")
    parser.add_argument("--views_per_object", type=int, default=1,
                        help="Number of views per object to show")
    parser.add_argument("--view_index", type=int, default=0,
                        help="Which view index to use (when views_per_object=1)")
    parser.add_argument(
        "--view_sampling", type=str, default="linspace",
        choices=["linspace", "consecutive"],
        help=(
            "How to choose views when views_per_object>1: "
            "'linspace' spreads across sequence, 'consecutive' picks nearby frames"
        ),
    )
    parser.add_argument(
        "--view_stride", type=int, default=1,
        help="Stride between selected views for consecutive sampling",
    )
    parser.add_argument(
        "--start_view", type=int, default=None,
        help=(
            "Starting view index for consecutive sampling; "
            "defaults to centered window when not provided"
        ),
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filter to specific CO3D categories (e.g., hydrant chair car)")
    parser.add_argument(
        "--black_threshold", type=float, default=0.01,
        help="Minimum mean pixel intensity in [0,1] to consider an image non-black",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_outputs/latent_pca_grid.png")
    args = parser.parse_args()

    if args.model_types is None:
        args.model_types = [None] * len(args.checkpoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.view_stride < 1:
        raise ValueError("--view_stride must be >= 1")
    if not (0.0 <= args.black_threshold <= 1.0):
        raise ValueError("--black_threshold must be in [0, 1]")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Import adapter inline to avoid heavy imports at top level
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_multiview_latent_consistency import NativeCO3DAdapter

    # Load adapter — request more objects than needed so we can filter by category
    n_request = 99999 if args.categories else args.num_objects
    print("Loading dataset adapter...")
    adapter = NativeCO3DAdapter(args.co3d_native_dir, n_request, args.seed)
    object_ids = adapter.get_object_ids()

    # Filter by category if specified
    if args.categories:
        object_ids = [oid for oid in object_ids if oid[0] in args.categories]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(object_ids)
        print(f"Filtered to categories {args.categories}: {len(object_ids)} sequences")

    n_views = args.views_per_object

    def select_view_indices(num_available):
        if n_views == 1:
            return [min(args.view_index, num_available - 1)]

        if args.view_sampling == "linspace":
            return np.linspace(0, num_available - 1, n_views, dtype=int).tolist()

        span = (n_views - 1) * args.view_stride + 1
        if num_available <= span:
            # Sequence too short: keep as close as possible and clip to last frame.
            return [
                min(i * args.view_stride, num_available - 1)
                for i in range(n_views)
            ]

        max_start = num_available - span
        if args.start_view is None:
            start = max_start // 2
        else:
            start = int(np.clip(args.start_view, 0, max_start))
        return [start + i * args.view_stride for i in range(n_views)]

    def _is_black(obj_id, view_idx):
        img_pil = adapter.load_view_image_pil(obj_id, view_idx, size=args.image_size)
        return np.asarray(img_pil, dtype=np.float32).mean() / 255.0 <= args.black_threshold

    def _replace_black_frames(obj_id, view_indices, num_available):
        """Replace black frames with the nearest non-black frame."""
        cleaned = []
        for idx in view_indices:
            if not _is_black(obj_id, idx):
                cleaned.append(idx)
                continue
            # Search outward for a non-black replacement
            found = False
            for offset in range(1, num_available):
                for candidate in [idx + offset, idx - offset]:
                    if 0 <= candidate < num_available and candidate not in cleaned:
                        if not _is_black(obj_id, candidate):
                            cleaned.append(candidate)
                            found = True
                            break
                if found:
                    break
            if not found:
                cleaned.append(idx)  # keep original if no replacement found
        return cleaned

    # Keep only sequences that have at least one non-black view.
    valid_object_ids = []
    for obj_id in object_ids:
        num_available = adapter.get_num_views(obj_id)
        view_indices = select_view_indices(num_available)
        cleaned = _replace_black_frames(obj_id, view_indices, num_available)
        # Reject only if ALL views are black
        if any(not _is_black(obj_id, vi) for vi in cleaned):
            valid_object_ids.append(obj_id)

    print(
        f"Valid non-black sequences: {len(valid_object_ids)} / {len(object_ids)} "
        f"(threshold={args.black_threshold})"
    )

    # Take only num_objects
    object_ids = valid_object_ids[:args.num_objects]
    n_objects = len(object_ids)
    print(f"Selected {n_objects} objects")

    if n_objects == 0:
        cats = f" in categories {args.categories}" if args.categories else ""
        print(f"ERROR: No valid (non-black) objects found{cats}. "
              "Try different --categories, --seed, or lower --black_threshold.")
        sys.exit(1)

    n_models = len(args.checkpoints)
    model_names = list(args.model_names)[:n_models]

    # Pre-load images (no GPU needed for this beyond the transform)
    print("\nLoading images...")
    all_images = [[] for _ in range(n_objects)]  # [obj_idx][view_idx]
    view_indices_per_obj = []
    for obj_idx, obj_id in enumerate(object_ids):
        num_available = adapter.get_num_views(obj_id)
        view_indices = select_view_indices(num_available)
        view_indices = _replace_black_frames(obj_id, view_indices, num_available)
        view_indices_per_obj.append(view_indices)
        for view_idx in view_indices:
            img = adapter.load_view_image(obj_id, view_idx, transform, device)
            all_images[obj_idx].append(img)
        print(
            f"  [{obj_idx+1}/{n_objects}] {adapter.get_object_name(obj_id)}: "
            f"{len(view_indices)} views (indices={view_indices})"
        )

    # Encode latents one model at a time to save GPU memory
    print("\nEncoding latents (sequential per model)...")
    # all_latents[model_idx][obj_idx][view_idx] = latent tensor (on CPU)
    all_latents = [[[] for _ in range(n_objects)] for _ in range(n_models)]

    for m_idx, (ckpt, cfg, name, mtype) in enumerate(
        zip(args.checkpoints, args.configs, args.model_names, args.model_types)
    ):
        print(f"\n  Loading {name}...")
        model, model_type = load_model(checkpoint_path=ckpt, config_path=cfg, model_type=mtype)
        model = model.to(device).eval()

        with torch.no_grad():
            for obj_idx in tqdm(range(n_objects), desc=f"  {name}", leave=True):
                for img in all_images[obj_idx]:
                    latent = encode_images(model, img, device, model_type)
                    all_latents[m_idx][obj_idx].append(latent.cpu())
        del model
        torch.cuda.empty_cache()

    # Compute consecutive cosine similarities per model per object
    print("\nComputing consecutive cosine similarities...")
    # cos_sims[m_idx][obj_idx] = list of n_views-1 cosine similarities
    cos_sims = [[[] for _ in range(n_objects)] for _ in range(n_models)]
    for m_idx in range(n_models):
        for obj_idx in range(n_objects):
            lats = all_latents[m_idx][obj_idx]
            for v in range(len(lats) - 1):
                flat_a = lats[v].flatten()
                flat_b = lats[v + 1].flatten()
                cs = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
                cos_sims[m_idx][obj_idx].append(cs)

    # Fit per-model PCA (1 component) so each model is normalized independently
    print("\nFitting per-model PCA...")
    pca_models = []
    per_model_ranges = []
    for m_idx in range(n_models):
        model_flat = []
        for obj_idx in range(n_objects):
            for lat in all_latents[m_idx][obj_idx]:
                l = lat[0].cpu().numpy()
                C, H, W = l.shape
                model_flat.append(l.reshape(C, -1).T)
        model_flat = np.vstack(model_flat)
        pca_m = PCA(n_components=1)
        pca_m.fit(model_flat)
        pc1 = pca_m.transform(model_flat)[:, 0]
        vmin, vmax = np.percentile(pc1, [2, 98])
        pca_models.append(pca_m)
        per_model_ranges.append((vmin, vmax))

    # Helper: apply PCA, return grayscale PC1 map
    def latent_to_pca_map(latent, pca_m, vmin, vmax):
        lat = latent[0] if latent.dim() == 4 else latent
        C, H, W = lat.shape
        lat_flat = lat.cpu().numpy().reshape(C, -1).T
        pc1 = pca_m.transform(lat_flat)[:, 0].reshape(H, W)
        pc1_norm = np.clip((pc1 - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return pc1_norm

    # Create figure
    n_cols = n_objects * n_views
    n_rows = 1 + n_models  # 1 row for images + 1 row per model
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for obj_idx in range(n_objects):
        for v_idx in range(n_views):
            col = obj_idx * n_views + v_idx

            # Row 0: original image
            img_np = denormalize(all_images[obj_idx][v_idx][0]).permute(1, 2, 0).cpu().numpy()
            axes[0, col].imshow(np.clip(img_np, 0, 1))
            category = object_ids[obj_idx][0]
            if n_views > 1:
                axes[0, col].set_title(f"view {v_idx}", fontsize=14, fontweight="bold")
            else:
                axes[0, col].set_title(category, fontsize=14, fontweight="bold")
            axes[0, col].axis("off")

            # Rows 1..n_models: PCA latents (per-model normalization)
            for m_idx in range(n_models):
                vmin, vmax = per_model_ranges[m_idx]
                lat_map = latent_to_pca_map(
                    all_latents[m_idx][obj_idx][v_idx], pca_models[m_idx],
                    vmin, vmax,
                )
                axes[1 + m_idx, col].imshow(lat_map, cmap="Spectral")
                axes[1 + m_idx, col].axis("off")

    plt.tight_layout(rect=[0.08, 0, 1, 1])  # leave space on left for labels

    # Row labels on column 0 using fig.text (after tight_layout so positions are final)
    for row, label in enumerate(["Input"] + model_names):
        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - 0.01, (bbox.y0 + bbox.y1) / 2, label,
            fontsize=18, fontweight="bold", ha="right", va="center", rotation=90
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
