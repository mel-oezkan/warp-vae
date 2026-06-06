#!/usr/bin/env python
"""
EQ-VAE latent interpolation demo on random CO3D categories.

For each randomly chosen CO3D sequence we pick a start frame and an end frame,
encode both with the EQ-VAE, linearly interpolate between the two latents at a
set of alphas, and decode every interpolated latent. We then plot, per sequence:

    Row 1: decoded RGB at each alpha   (start | interp ... | end)
    Row 2: PCA-RGB of each latent      (shared PCA fit across the row)

This reads the *raw* CO3D layout (per-category ``frame_annotations.jgz`` with
``image``/``mask``/``viewpoint`` fields), so it works on the held-out
``co3d_data`` root that is not covered by the training bb_file annotations.

Runs on CPU by default to avoid disturbing GPU training runs.

Example:
    python scripts/demos/eqvae_latent_interp_demo.py \
        --checkpoint "checkpoints/natural-illegal-bullmastiff-.../last.ckpt" \
        --config config/eqvae_hydrant_cropped.yaml \
        --data_root /visinf/projects_students/dlcv2025_groupZ/co3d_data \
        --num_sequences 4 --num_steps 5 --seed 0
"""

import argparse
import gzip
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.analysis import load_model, encode_images, decode_latents, latent_to_pca_rgb
from src.analysis.model_utils import denormalize

# Keep torch.compile / dynamo out of the way (matches compare_latents.py).
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


def patch_attention_for_cpu():
    """Replace the xformers-based AttnBlock forward with a CPU-friendly SDPA.

    The VAE's mid-block self-attention uses ``xformers.memory_efficient_attention``,
    which only supports CUDA. When we run on CPU (to avoid disturbing GPU training
    runs) we swap in ``torch.nn.functional.scaled_dot_product_attention``, which is
    numerically equivalent single-head self-attention.
    """
    import torch.nn.functional as F
    from einops import rearrange
    from ldm.modules.diffusionmodules import model as ldm_model

    def cpu_forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        B, C, H, W = q.shape
        q, k, v = (rearrange(t, "b c h w -> b (h w) c") for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # (B, HW, C)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return x + self.proj_out(out)

    ldm_model.MemoryEfficientAttnBlock.forward = cpu_forward


def list_categories(data_root: Path):
    """Categories that have at least one extracted sequence (a dir with images/)."""
    cats = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "frame_annotations.jgz").exists():
            continue
        has_seq = any((s / "images").is_dir() for s in d.iterdir() if s.is_dir())
        if has_seq:
            cats.append(d.name)
    return cats


def load_frame_annotations(data_root: Path, category: str):
    """Return {sequence_name: [frame_ann, ...]} for frames whose image exists on disk."""
    fa = data_root / category / "frame_annotations.jgz"
    with gzip.GzipFile(fa, "rb") as f:
        anns = json.loads(f.read().decode("utf8"))

    seqs = {}
    for a in anns:
        img_rel = a["image"]["path"]
        if not (data_root / img_rel).exists():
            continue
        seqs.setdefault(a["sequence_name"], []).append(a)
    # keep frame order stable
    for s in seqs:
        seqs[s].sort(key=lambda a: a["frame_number"])
    return {s: fr for s, fr in seqs.items() if len(fr) >= 2}


def mask_bbox(mask_path: Path, pad_frac: float = 0.05):
    """Square bbox (x1, y1, x2, y2) around the foreground mask, or None."""
    if not mask_path.exists():
        return None
    m = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(m > 10)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2
    half *= (1.0 + pad_frac)
    return np.array([cx - half, cy - half, cx + half, cy + half])


def load_image(data_root: Path, ann: dict, image_size: int, crop: bool):
    """Load a frame, optionally crop to its mask bbox, resize, normalize to [-1, 1]."""
    img = Image.open(data_root / ann["image"]["path"]).convert("RGB")
    w, h = img.size

    if crop:
        bbox = mask_bbox(data_root / ann["mask"]["path"]) if "mask" in ann else None
        if bbox is None:
            side = min(w, h)
            bbox = np.array([(w - side) / 2, (h - side) / 2,
                             (w + side) / 2, (h + side) / 2])
        bbox = np.around(bbox).astype(int)
        img = transforms.functional.crop(
            img, top=bbox[1], left=bbox[0],
            height=bbox[3] - bbox[1], width=bbox[2] - bbox[0],
        )

    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img)


def plot_sequence(images_rgb, latents_rgb, alphas, title, save_path):
    """Two-row grid: decoded RGB (top), latent PCA-RGB (bottom)."""
    n = len(alphas)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, a in enumerate(alphas):
        if j == 0:
            tag = "start (a=0.00)"
        elif j == n - 1:
            tag = "end (a=1.00)"
        else:
            tag = f"a={a:.2f}"

        axes[0, j].imshow(images_rgb[j])
        axes[0, j].set_title(tag, fontsize=9)
        axes[0, j].axis("off")

        axes[1, j].imshow(latents_rgb[j])
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("decoded", fontsize=10)
    axes[1, 0].set_ylabel("latent (PCA-RGB)", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default="/visinf/projects_students/dlcv2025_groupZ/co3d_data")
    p.add_argument("--output_dir", default="eval_outputs/eqvae_latent_interp")
    p.add_argument("--num_sequences", type=int, default=4,
                   help="How many random (category, sequence) pairs to interpolate.")
    p.add_argument("--num_steps", type=int, default=5,
                   help="Number of interpolation points incl. start and end (>=2).")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--no_crop", action="store_true",
                   help="Disable mask-bbox cropping (use center crop).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="cpu (default, safe for shared GPUs) or cuda.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cpu":
        patch_attention_for_cpu()

    print("\nLoading EQ-VAE...")
    model, model_type = load_model(args.checkpoint, args.config, model_type="auto")
    model = model.to(device).eval()
    print(f"Model type: {model_type}")

    cats = list_categories(data_root)
    print(f"\nFound {len(cats)} categories with extracted sequences.")
    random.shuffle(cats)

    alphas = np.linspace(0.0, 1.0, args.num_steps)
    crop = not args.no_crop

    made = 0
    tried_cats = iter(cats)
    while made < args.num_sequences:
        try:
            cat = next(tried_cats)
        except StopIteration:
            print("Ran out of categories before reaching requested count.")
            break

        seqs = load_frame_annotations(data_root, cat)
        if not seqs:
            continue
        seq_name = random.choice(list(seqs.keys()))
        frames = seqs[seq_name]

        # Random distinct start/end; bias toward a wider gap for a visible change.
        i0 = random.randint(0, len(frames) - 2)
        i1 = random.randint(i0 + 1, len(frames) - 1)
        f0, f1 = frames[i0], frames[i1]

        print(f"\n[{made + 1}/{args.num_sequences}] {cat}/{seq_name} "
              f"frames {f0['frame_number']} -> {f1['frame_number']} "
              f"({len(frames)} frames on disk)")

        img0 = load_image(data_root, f0, args.image_size, crop)
        img1 = load_image(data_root, f1, args.image_size, crop)
        imgs = torch.stack([img0, img1])

        z = encode_images(model, imgs, device, model_type)  # (2, C, h, w)
        z0, z1 = z[0:1], z[1:2]

        # Linear interpolation in latent space.
        z_interp = torch.cat([(1 - a) * z0 + a * z1 for a in alphas], dim=0)
        recon = decode_latents(model, z_interp, device, model_type)  # (n, 3, H, W)

        images_rgb = [denormalize(recon[j]).permute(1, 2, 0).cpu().numpy()
                      for j in range(len(alphas))]

        # Shared PCA across the row so colors are comparable along the path.
        all_lat = z_interp.reshape(z_interp.shape[0], z_interp.shape[1], -1)
        flat = all_lat.permute(0, 2, 1).reshape(-1, z_interp.shape[1]).cpu().numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3).fit(flat)
        latents_rgb = [latent_to_pca_rgb(z_interp[j], pca_model=pca)[0]
                       for j in range(len(alphas))]

        title = (f"EQ-VAE latent interpolation — {cat}/{seq_name}  "
                 f"(frame {f0['frame_number']} -> {f1['frame_number']})")
        save_path = out_dir / f"interp_{made:02d}_{cat}_{seq_name}.png"
        plot_sequence(images_rgb, latents_rgb, alphas, title, save_path)
        print(f"  saved {save_path}")
        made += 1

    print(f"\nDone. {made} interpolation figure(s) in {out_dir}/")


if __name__ == "__main__":
    main()
