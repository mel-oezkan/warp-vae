"""Compare different pair-selection measures for warp-based training.

For each of a few random CO3D sequences, picks one anchor frame and ranks all
other frames in the sequence by several measures:

    M1  position L2                  (current baseline)
    M2  weighted position + rotation (naive idea)
    M3  object-centric angle         (angle at scene look-at center)
    M4  co-visibility (sphere proxy) (IoU of projected scene-bound sphere)
    M5  RoMA mean overlap confidence (reference: what the warp will actually do)

Builds one figure per sequence: rows = measures, columns = anchor + top-K picks.
Each picked frame shows its score under EVERY measure, so you can read across
rows to see where the measures agree or disagree. A second figure shows the
RoMA confidence map for each picked pair.

Usage:
    python scripts/sweeps/compare_pair_metrics.py \\
        --annotation_path /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \\
        --co3d_root /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --output_dir eval_outputs/compare_pair_metrics \\
        --num_sequences 3 --top_k 4
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.camera_utils import (
    extract_co3d_camera_positions,
    load_co3d_annotations,
)
from src.analysis.roma_metrics import load_roma_model


# ---------------------------------------------------------------------------
# Pair-selection measures
# ---------------------------------------------------------------------------

def estimate_scene_center(positions: np.ndarray, forwards: np.ndarray) -> np.ndarray:
    """Least-squares point closest to all camera viewing rays."""
    D = forwards / (np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-8)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, D):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    return np.linalg.solve(A + 1e-6 * np.eye(3), b)


def camera_forwards(frames: list) -> np.ndarray:
    """Camera forward direction in world frame. CO3D: R is world->cam, so
    camera +Z in world is R.T @ [0,0,1] (= R's third row)."""
    return np.array([np.array(f["R"])[2, :] for f in frames])


def m1_position_l2(positions: np.ndarray, anchor: int) -> np.ndarray:
    return np.linalg.norm(positions - positions[anchor], axis=1)


def m2_weighted_pos_rot(
    frames: list, positions: np.ndarray, anchor: int,
    w_trans: float = 1.0, w_rot: float = 1.0,
) -> np.ndarray:
    Rs = [np.array(f["R"]) for f in frames]
    R_a = Rs[anchor]
    trans = np.linalg.norm(positions - positions[anchor], axis=1)
    rot = np.zeros(len(frames))
    for j, Rj in enumerate(Rs):
        R_rel = R_a @ Rj.T
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot[j] = np.degrees(np.arccos(tr))
    # Normalize each component by its max for fair weighting
    trans_n = trans / (trans.max() + 1e-8)
    rot_n = rot / (rot.max() + 1e-8)
    return w_trans * trans_n + w_rot * rot_n


def m3_object_centric_angle(
    positions: np.ndarray, anchor: int, center: np.ndarray,
) -> np.ndarray:
    """Angle (deg) subtended at scene center between camera positions."""
    v = positions - center
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    cos = np.clip(v_norm @ v_norm[anchor], -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def m4_covis_sphere(
    frames: list, positions: np.ndarray, anchor: int,
    center: np.ndarray, n_samples: int = 200, seed: int = 0,
) -> np.ndarray:
    """Co-visibility proxy: sample points on a sphere around the object
    center (radius = median distance from center to cameras / 4 as a rough
    object size), project into both cameras using CO3D NDC convention,
    measure IoU of in-frame projections.

    NDC projection: x_ndc = f_x * x_cam / z_cam + p_x. In-frame if |x_ndc|<=1
    and z_cam > 0 (point in front of camera). For a 1:1 aspect, both axes
    use [-1, 1].
    """
    rng = np.random.default_rng(seed)
    # Estimate object radius from cameras' distance to center
    cam_radii = np.linalg.norm(positions - center, axis=1)
    radius = float(np.median(cam_radii)) * 0.25  # rough object extent

    # Uniform points on a sphere
    pts = rng.normal(size=(n_samples, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts = center + radius * pts  # (N, 3) world

    def visible_mask(frame):
        R = np.array(frame["R"])  # world -> cam
        T = np.array(frame["T"])
        fx, fy = frame["focal_length"]
        px, py = frame["principal_point"]
        cam = pts @ R.T + T  # (N, 3) in camera frame
        z = cam[:, 2]
        in_front = z > 1e-3
        x_ndc = fx * cam[:, 0] / np.where(in_front, z, 1.0) + px
        y_ndc = fy * cam[:, 1] / np.where(in_front, z, 1.0) + py
        return in_front & (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)

    masks = np.array([visible_mask(f) for f in frames])  # (F, N)
    anchor_mask = masks[anchor]
    iou = np.zeros(len(frames))
    for j in range(len(frames)):
        inter = (anchor_mask & masks[j]).sum()
        union = (anchor_mask | masks[j]).sum()
        iou[j] = inter / max(union, 1)
    # Return as a distance (lower = more similar), so higher IoU -> lower score
    return 1.0 - iou


# ---------------------------------------------------------------------------
# RoMA reference
# ---------------------------------------------------------------------------

def load_and_crop(filepath: Path, bbox: list, resolution: int) -> Image.Image:
    """Crop with square_bbox padding (same convention as precomputed warps)
    and resize to `resolution`."""
    from data_process.co3d_dataset import square_bbox
    img = Image.open(filepath).convert("RGB")
    sq = np.around(square_bbox(np.array(bbox))).astype(int)
    x1, y1, x2, y2 = [int(v) for v in sq]
    img = img.crop((x1, y1, x2, y2))
    return img.resize((resolution, resolution), Image.LANCZOS)


@torch.no_grad()
def roma_mean_confidence(roma, img_a: Image.Image, img_b: Image.Image):
    """Return (mean confidence, full HxW conf map as numpy)."""
    pred = roma.match(img_a, img_b)
    warp = pred["warp_AB"]  # (1, H, W, 2)
    overlap = pred.get("overlap_AB")
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)
    in_bounds = (warp.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    conf = torch.clamp(overlap * in_bounds, 0.0, 1.0)[0, ..., 0]  # (H, W)
    return float(conf.mean().item()), conf.cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MEASURE_NAMES = ["pos L2", "pos+rot", "obj-angle", "1-covis", "1-RoMA"]


def make_figure(
    seq_name: str,
    anchor_idx: int,
    frame_paths: list,
    bboxes: list,
    scores: dict,  # name -> np.array (n_frames,)
    top_picks: dict,  # name -> list[int] of length K
    roma_conf_maps: dict,  # (anchor, j) -> 2D array, optional
    save_path: Path,
    resolution: int = 192,
    top_k: int = 4,
):
    """One row per measure; columns: anchor + K picks. Caption per cell
    lists this frame's score under every measure (so we can see
    cross-measure agreement)."""
    n_rows = len(MEASURE_NAMES)
    n_cols = top_k + 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    def load_thumb(idx):
        return np.array(load_and_crop(frame_paths[idx], bboxes[idx], resolution))

    anchor_thumb = load_thumb(anchor_idx)

    for ri, mname in enumerate(MEASURE_NAMES):
        # Anchor column
        ax = axes[ri][0]
        ax.imshow(anchor_thumb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(mname, fontsize=11, fontweight="bold")
        if ri == 0:
            ax.set_title(f"anchor f{anchor_idx}", fontsize=9)

        # Pick columns
        picks = top_picks[mname]
        for ci, j in enumerate(picks):
            ax = axes[ri][ci + 1]
            ax.imshow(load_thumb(j))
            ax.set_xticks([]); ax.set_yticks([])
            # Caption: this frame's value under EACH measure (own measure bolded)
            lines = [f"f{j}"]
            for nm in MEASURE_NAMES:
                v = scores[nm][j]
                tag = f"{nm}={v:.3f}"
                if nm == mname:
                    tag = "*" + tag + "*"
                lines.append(tag)
            ax.set_title("\n".join(lines), fontsize=7)

    fig.suptitle(
        f"{seq_name}  |  anchor=f{anchor_idx}  |  rows=measure, cols=that measure's top-{top_k}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_roma_conf_figure(
    seq_name: str,
    anchor_idx: int,
    top_picks: dict,
    roma_conf_maps: dict,
    save_path: Path,
    top_k: int = 4,
):
    """Same layout as the main figure but each cell shows the RoMA
    confidence map (anchor->candidate) instead of the RGB thumbnail.
    Anchor column is blank."""
    n_rows = len(MEASURE_NAMES)
    n_cols = top_k + 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )
    for ri, mname in enumerate(MEASURE_NAMES):
        ax = axes[ri][0]
        ax.axis("off")
        ax.set_ylabel(mname, fontsize=11, fontweight="bold")
        if ri == 0:
            ax.set_title("anchor", fontsize=9)
        picks = top_picks[mname]
        for ci, j in enumerate(picks):
            ax = axes[ri][ci + 1]
            cmap = roma_conf_maps.get((anchor_idx, j))
            if cmap is None:
                ax.axis("off")
                continue
            im = ax.imshow(cmap, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"f{j}  mean={cmap.mean():.3f}", fontsize=8)

    fig.suptitle(
        f"{seq_name}  |  anchor=f{anchor_idx}  |  RoMA overlap confidence (anchor→pick)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotation_path",
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz",
    )
    ap.add_argument(
        "--co3d_root",
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full",
    )
    ap.add_argument("--output_dir", default="eval_outputs/compare_pair_metrics")
    ap.add_argument("--num_sequences", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--max_frames_per_seq", type=int, default=80,
                    help="Subsample frames in a sequence to bound RoMA cost.")
    ap.add_argument("--roma_setting", default="turbo",
                    choices=["precise", "fast", "turbo", "base"])
    ap.add_argument("--w_trans", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    co3d_root = Path(args.co3d_root)

    print(f"Loading annotations from {args.annotation_path}")
    annotations = load_co3d_annotations(args.annotation_path)
    all_seqs = list(annotations.keys())
    seq_names = random.sample(all_seqs, args.num_sequences)
    print(f"Selected sequences: {seq_names}")

    print(f"Loading RoMA ({args.roma_setting}) on {args.device}")
    roma = load_roma_model(setting=args.roma_setting, device=args.device, compile=False)

    for seq_name in seq_names:
        frames_full = annotations[seq_name]
        if len(frames_full) < args.top_k + 2:
            print(f"  skipping {seq_name}: too few frames")
            continue

        # Subsample evenly to bound cost
        if len(frames_full) > args.max_frames_per_seq:
            stride = len(frames_full) / args.max_frames_per_seq
            keep = sorted({int(i * stride) for i in range(args.max_frames_per_seq)})
            frames = [frames_full[i] for i in keep]
            orig_indices = keep
        else:
            frames = frames_full
            orig_indices = list(range(len(frames_full)))

        n = len(frames)
        positions = extract_co3d_camera_positions(frames)
        forwards = camera_forwards(frames)
        center = estimate_scene_center(positions, forwards)

        # Pick anchor (deterministic per seq)
        anchor = random.randrange(n)

        print(f"\n=== {seq_name} | n_frames={n} | anchor=f{anchor} "
              f"(orig idx {orig_indices[anchor]}) ===")
        print(f"  scene center estimate: {center}, "
              f"||center||={np.linalg.norm(center):.3f}")

        # --- Cheap measures ---
        scores = {
            "pos L2":    m1_position_l2(positions, anchor),
            "pos+rot":   m2_weighted_pos_rot(frames, positions, anchor,
                                             args.w_trans, args.w_rot),
            "obj-angle": m3_object_centric_angle(positions, anchor, center),
            "1-covis":   m4_covis_sphere(frames, positions, anchor, center),
        }

        # --- RoMA reference: run anchor vs all other frames ---
        frame_paths = [co3d_root / f["filepath"] for f in frames]
        bboxes = [f["bbox"] for f in frames]

        anchor_img = load_and_crop(frame_paths[anchor], bboxes[anchor],
                                   args.image_size)
        roma_scores = np.zeros(n)
        roma_conf_maps = {}
        for j in range(n):
            if j == anchor:
                roma_scores[j] = 0.0  # placeholder; will be excluded from ranking
                continue
            img_j = load_and_crop(frame_paths[j], bboxes[j], args.image_size)
            mean_conf, conf_map = roma_mean_confidence(roma, anchor_img, img_j)
            roma_scores[j] = mean_conf
            roma_conf_maps[(anchor, j)] = conf_map
            if j % 10 == 0:
                print(f"    RoMA {j}/{n}  mean_conf={mean_conf:.3f}")
        # Distance form: 1 - mean confidence (lower = better)
        scores["1-RoMA"] = 1.0 - roma_scores

        # --- Top-K picks per measure (excluding the anchor itself) ---
        top_picks = {}
        for name, sc in scores.items():
            order = np.argsort(sc)
            picks = [int(i) for i in order if i != anchor][: args.top_k]
            top_picks[name] = picks

        # --- Save figures ---
        fig_path = out_dir / f"{seq_name}_anchor_f{anchor}.png"
        make_figure(
            seq_name, anchor, frame_paths, bboxes,
            scores, top_picks, roma_conf_maps,
            fig_path, resolution=args.image_size, top_k=args.top_k,
        )
        print(f"  wrote {fig_path}")

        conf_fig_path = out_dir / f"{seq_name}_anchor_f{anchor}_roma_conf.png"
        make_roma_conf_figure(
            seq_name, anchor, top_picks, roma_conf_maps,
            conf_fig_path, top_k=args.top_k,
        )
        print(f"  wrote {conf_fig_path}")

        # Also print a small numeric summary: for each measure's top-K picks,
        # what's the mean RoMA confidence? (= how good the warps will be)
        print("  Mean RoMA confidence of each measure's top-K picks:")
        for name, picks in top_picks.items():
            mean_conf = float(np.mean([roma_scores[j] for j in picks]))
            print(f"    {name:>10s}: {mean_conf:.3f}  (picks={picks})")

    print(f"\nAll figures in {out_dir}")


if __name__ == "__main__":
    main()
