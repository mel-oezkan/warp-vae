"""For each CO3D sequence, plot 3 frames spaced N steps apart in a single figure.

Usage:
    python scripts/visualization/plot_sequence_frames.py
    python scripts/visualization/plot_sequence_frames.py --step 10 --num_sequences 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.camera_utils import load_co3d_annotations


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_path", type=str,
                   default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz")
    p.add_argument("--image_root", type=str,
                   default="/visinf/projects_students/dlcv2025_groupZ/co3d_full")
    p.add_argument("--output_dir", type=str, default="eval_outputs/camera/hydrant/frame_triplets")
    p.add_argument("--num_sequences", type=int, default=50, help="-1 for all")
    p.add_argument("--step", type=int, default=10, help="Frame spacing between the 3 shown frames")
    p.add_argument("--start", type=int, default=0, help="Index of first frame to show")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    annotations = load_co3d_annotations(args.annotation_path)
    seq_names = list(annotations.keys())
    if args.num_sequences > 0:
        seq_names = seq_names[: args.num_sequences]

    saved = 0
    for name in seq_names:
        frames = annotations[name]
        idxs = [args.start, args.start + args.step, args.start + 2 * args.step]
        if idxs[-1] >= len(frames):
            # fall back to evenly spaced if step too large
            idxs = list(np.linspace(0, len(frames) - 1, 3).astype(int))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        ok = True
        for ax, i in zip(axes, idxs):
            img_path = image_root / frames[i]["filepath"]
            if not img_path.exists():
                ax.text(0.5, 0.5, f"missing\n{img_path.name}", ha="center", va="center")
                ax.set_axis_off()
                ok = False
                continue
            img = Image.open(img_path)
            ax.imshow(np.asarray(img))
            ax.set_title(f"frame {i}", fontsize=10)
            ax.set_axis_off()

        fig.suptitle(f"{name}  |  frames {idxs[0]}, {idxs[1]}, {idxs[2]}  (of {len(frames)})",
                     fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"{name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved += 1
        if not ok:
            print(f"  (some images missing for {name})")

    print(f"Saved {saved} sequence triplet figures to {out_dir}/")


if __name__ == "__main__":
    main()
