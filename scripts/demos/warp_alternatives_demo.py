"""
Demo: alternatives to noisy inter-frame warps for VAE consistency losses.

We synthesize a tiny 3D scene (a textured fronto-parallel "wall" plus a
foreground "box" floating in front of it) and render it from two camera
poses. From these we derive four candidate warps that map frame A -> frame B,
each with progressively less mathematical structure:

  1. SE(3) / rotation-only equivariance (group, exact inverse)
       - Camera rotates in place. The mapping pixel-A -> pixel-B is a
         homography that depends only on the rotation. Bijective everywhere.

  2. Homography (group, exact inverse, but breaks under parallax)
       - We fit a single H to the *plane* and apply it everywhere.
         Works perfectly on the wall, wrong on the box.

  3. Depth + pose reprojection (derived from a group action on 3D points)
       - For every pixel we unproject using ground-truth depth, transform
         by the camera motion, and reproject. Occlusions become a
         *computable* mask, not a heuristic.

  4. RoMA-style noisy warp (no group structure, no exact inverse)
       - Same depth-reprojection corrupted with Gaussian flow noise and a
         random missing region, mimicking what a learned matcher returns.

For each warp we visualise, on a common pair of frames:
    (a) the warped frame A,
    (b) the validity mask (where the warp is trustworthy),
    (c) the residual |W(A) - B| inside the valid region.

Output: outputs/scripts/warp_alternatives_demo.png

Run:
    conda activate cv
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/warp_alternatives_demo.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


H, W = 192, 256
FOCAL = 220.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1.0]])
K_inv = np.linalg.inv(K)


# ---------------------------------------------------------------------------
# Scene + rendering
# ---------------------------------------------------------------------------
def make_texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A high-frequency checker + colour gradient so warp errors are visible."""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    checker = ((xs // 16 + ys // 16) % 2).astype(np.float32)
    grad_r = (xs / w).astype(np.float32)
    grad_g = (ys / h).astype(np.float32)
    grad_b = ((xs + ys) / (w + h)).astype(np.float32)
    noise = rng.uniform(0, 0.15, size=(h, w)).astype(np.float32)
    img = np.stack([0.4 * checker + 0.6 * grad_r,
                    0.4 * checker + 0.6 * grad_g,
                    0.4 * checker + 0.6 * grad_b], axis=-1)
    img = np.clip(img + noise[..., None], 0, 1)
    return img


def render_scene(R: np.ndarray, t: np.ndarray):
    """
    Render a scene composed of:
      - a back wall at Z = Z_wall (fronto-parallel, textured)
      - a foreground box at Z = Z_box (smaller, offset, different texture)

    Camera extrinsics are world->camera: x_cam = R x_world + t.
    Returns (image HxWx3, depth HxW, fg_mask HxW bool).
    """
    Z_wall, Z_box = 6.0, 3.0
    img = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    fg_mask = np.zeros((H, W), dtype=bool)

    wall_tex = make_texture(H, W, seed=1)
    box_tex = make_texture(H // 2, W // 2, seed=7)

    # Each pixel ray in cam frame
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
    rays_cam = pix @ K_inv.T                                      # HxWx3

    # Back-project to world: x_world = R^T (Z * d_cam - t)
    R_inv = R.T

    # Render wall: world plane Z_world = Z_wall (defined in world frame).
    # Find lambda so that (R_inv (lambda d - t))_z = Z_wall.
    # Let a = R_inv d, b = R_inv t.  Then a_z * lambda - b_z = Z_wall
    # => lambda = (Z_wall + b_z) / a_z.
    a = rays_cam @ R_inv.T                                        # HxWx3
    b = R_inv @ t
    lam_wall = (Z_wall + b[2]) / (a[..., 2] + 1e-9)
    P_cam_wall = lam_wall[..., None] * rays_cam                   # HxWx3
    P_world_wall = (P_cam_wall - t) @ R_inv.T                     # HxWx3
    # Sample wall texture by world (X, Y) coords.
    u = ((P_world_wall[..., 0] + 4) / 8 * W).astype(np.int32)
    v = ((P_world_wall[..., 1] + 3) / 6 * H).astype(np.int32)
    valid_wall = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (lam_wall > 0)
    img_wall = np.zeros_like(img)
    img_wall[valid_wall] = wall_tex[v[valid_wall], u[valid_wall]]
    depth_wall = np.where(valid_wall, lam_wall * rays_cam[..., 2] / (rays_cam[..., 2] + 1e-9), np.inf)
    # depth_wall is lam_wall * (z-component of unit ray in cam) -> just lam_wall * 1 because rays have z=1.
    depth_wall = np.where(valid_wall, lam_wall, np.inf)

    # Render box: a quad in world coords parallel to wall but at Z_box, occupying
    # world X in [-1, 1], Y in [-0.8, 0.8].
    lam_box = (Z_box + b[2]) / (a[..., 2] + 1e-9)
    P_cam_box = lam_box[..., None] * rays_cam
    P_world_box = (P_cam_box - t) @ R_inv.T
    in_box = (np.abs(P_world_box[..., 0]) < 1.0) & (np.abs(P_world_box[..., 1]) < 0.8) & (lam_box > 0)
    u_b = ((P_world_box[..., 0] + 1) / 2 * (W // 2)).astype(np.int32)
    v_b = ((P_world_box[..., 1] + 0.8) / 1.6 * (H // 2)).astype(np.int32)
    u_b = np.clip(u_b, 0, W // 2 - 1)
    v_b = np.clip(v_b, 0, H // 2 - 1)
    img_box = np.zeros_like(img)
    img_box[in_box] = box_tex[v_b[in_box], u_b[in_box]]
    depth_box = np.where(in_box, lam_box, np.inf)

    # Composite: nearer surface wins.
    use_box = in_box & (depth_box < depth_wall)
    img = np.where(use_box[..., None], img_box, img_wall)
    depth = np.where(use_box, depth_box, depth_wall)
    fg_mask = use_box
    return img, depth, fg_mask


def rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------
# Warps
# ---------------------------------------------------------------------------
def warp_image(img: np.ndarray, flow_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Backward warp: out[y, x] = img[ flow_xy[y, x, 1], flow_xy[y, x, 0] ].
    `flow_xy` gives, for each *target* pixel in frame B, the (x, y) it samples
    from frame A.  Out-of-bounds or invalid pixels become 0.
    """
    h, w = img.shape[:2]
    xs = np.round(flow_xy[..., 0]).astype(np.int32)
    ys = np.round(flow_xy[..., 1]).astype(np.int32)
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & valid
    out = np.zeros_like(img)
    out[in_bounds] = img[ys[in_bounds], xs[in_bounds]]
    return out, in_bounds


def warp_rotation_only(R_a: np.ndarray, R_b: np.ndarray):
    """
    Camera rotates in place between A and B (t identical).  Then
    pixel_B = K R_b R_a^T K^{-1} pixel_A.  Exact, bijective, group element.
    Returns the *backward* flow: for each B-pixel, where does it come from in A.
    """
    H_mat = K @ R_a @ R_b.T @ K_inv          # B -> A
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix_b = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
    pix_a = pix_b @ H_mat.T
    pix_a = pix_a[..., :2] / (pix_a[..., 2:3] + 1e-9)
    return pix_a, np.ones((H, W), dtype=bool)


def warp_homography_plane(R_a, t_a, R_b, t_b, plane_n_world, plane_d_world):
    """
    Homography induced by the back-plane only.  Exact on the wall, *wrong*
    on the box (parallax).  Still a group element.
    """
    # Plane in camera A:  n_a^T X_a = d_a, where X_a = R_a X_w + t_a, so
    # n_w^T X_w = d_w  =>  n_w^T R_a^T (X_a - t_a) = d_w
    # => (R_a n_w)^T X_a = d_w + n_w^T R_a^T t_a   ... but cleaner to just
    # build using both extrinsics directly:
    # H_{B<-A} = K (R_ba - t_ba n_a^T / d_a) K^{-1}
    R_ba = R_b @ R_a.T
    t_ba = t_b - R_ba @ t_a
    # plane in A coords
    n_a = R_a @ plane_n_world
    d_a = plane_d_world - n_a @ t_a   # because n_w X_w = d -> n_a (X_a - t_a) = d  (n_w in world, transformed)
    # Be safe: recompute d_a from a known world point on the plane.
    pt_w = plane_n_world * plane_d_world           # any point with n_w^T x = d_w when |n|=1
    pt_a = R_a @ pt_w + t_a
    d_a = n_a @ pt_a
    H_ba = K @ (R_ba - np.outer(t_ba, n_a) / d_a) @ K_inv
    H_ab = np.linalg.inv(H_ba)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix_b = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
    pix_a = pix_b @ H_ab.T
    pix_a = pix_a[..., :2] / (pix_a[..., 2:3] + 1e-9)
    return pix_a, np.ones((H, W), dtype=bool)


def warp_depth_reproject(depth_b, R_a, t_a, R_b, t_b):
    """
    For each pixel in B, unproject using B's depth, transform to world, then
    to A's camera, project into A.  Geometry-derived; not a group element,
    but failure modes are *predictable*: it's wrong only where B's depth is
    wrong, and occlusions can be detected by a forward-backward check.
    """
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix_b = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)
    rays_b = pix_b @ K_inv.T
    finite = np.isfinite(depth_b)
    P_cam_b = rays_b * np.where(finite, depth_b, 0)[..., None]
    # cam_b -> world -> cam_a
    P_world = (P_cam_b - t_b) @ R_b
    P_cam_a = P_world @ R_a.T + t_a
    pix_a_h = P_cam_a @ K.T
    z = pix_a_h[..., 2:3]
    pix_a = pix_a_h[..., :2] / (z + 1e-9)
    valid = finite & (z[..., 0] > 0)
    return pix_a, valid


def corrupt_to_roma(flow_xy, valid, seed=0):
    """Add Gaussian flow noise + a random circular dropout to mimic a learned matcher."""
    rng = np.random.default_rng(seed)
    noisy = flow_xy + rng.normal(0, 3.0, size=flow_xy.shape).astype(np.float32)
    cy = rng.integers(H // 4, 3 * H // 4)
    cx = rng.integers(W // 4, 3 * W // 4)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dropout = (xs - cx) ** 2 + (ys - cy) ** 2 < 30 ** 2
    valid = valid & ~dropout
    return noisy, valid


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main():
    out_path = "/visinf/home/lab_mozkan/computer-vision-proj-lab/outputs/scripts/warp_alternatives_demo.png"

    # Common scene: wall at Z=6 (normal -z in world), box at Z=3.
    # Two cameras with both rotation and translation -> real parallax for the box.
    theta_a, theta_b = -0.08, 0.08
    R_a = rotation_y(theta_a)
    R_b = rotation_y(theta_b)
    t_a = np.array([-0.25, 0.0, 0.0])
    t_b = np.array([+0.25, 0.0, 0.0])
    # Also a "rotation-only" pair, no translation:
    R_a_ro = rotation_y(-0.1)
    R_b_ro = rotation_y(+0.1)
    t_ro = np.zeros(3)

    img_a_ro, _, _ = render_scene(R_a_ro, t_ro)
    img_b_ro, _, _ = render_scene(R_b_ro, t_ro)
    img_a, depth_a, fg_a = render_scene(R_a, t_a)
    img_b, depth_b, fg_b = render_scene(R_b, t_b)

    # 1) Rotation-only (group).
    flow_ro, val_ro = warp_rotation_only(R_a_ro, R_b_ro)
    w_ro, m_ro = warp_image(img_a_ro, flow_ro, val_ro)
    res_ro = np.abs(w_ro - img_b_ro).mean(-1) * m_ro

    # 2) Homography of the back plane (group, breaks on box).
    plane_n_world = np.array([0.0, 0.0, 1.0])   # the back wall is Z = 6
    plane_d_world = 6.0
    flow_h, val_h = warp_homography_plane(R_a, t_a, R_b, t_b, plane_n_world, plane_d_world)
    w_h, m_h = warp_image(img_a, flow_h, val_h)
    res_h = np.abs(w_h - img_b).mean(-1) * m_h

    # 3) Depth + pose reprojection (geometry-derived, occlusion-aware).
    flow_d, val_d = warp_depth_reproject(depth_b, R_a, t_a, R_b, t_b)
    w_d, m_d = warp_image(img_a, flow_d, val_d)
    res_d = np.abs(w_d - img_b).mean(-1) * m_d

    # 4) RoMA-like noisy warp (same geometry, corrupted).
    flow_n, val_n = corrupt_to_roma(flow_d, val_d, seed=2)
    w_n, m_n = warp_image(img_a, flow_n, val_n)
    res_n = np.abs(w_n - img_b).mean(-1) * m_n

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(5, 5, figure=fig, hspace=0.35, wspace=0.1)

    # Row 0: source / target reminder.
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_a_ro); ax.set_title("Frame A (rot-only)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(img_b_ro); ax.set_title("Frame B (rot-only)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(img_a); ax.set_title("Frame A (with translation)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(img_b); ax.set_title("Frame B (with translation)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 4]); ax.imshow(depth_b, cmap="magma"); ax.set_title("Depth(B)"); ax.axis("off")

    rows = [
        ("1. Rotation-only (group)",          img_a_ro, img_b_ro, w_ro, m_ro, res_ro,
         "Bijective. Exact inverse. The constraint EQ-VAE relies on."),
        ("2. Homography (plane-only)",        img_a,    img_b,    w_h,  m_h,  res_h,
         "Group element, but wrong on the foreground box (parallax)."),
        ("3. Depth + pose reprojection",      img_a,    img_b,    w_d,  m_d,  res_d,
         "Not a group; but failure modes are computable from depth."),
        ("4. RoMA-like noisy warp",           img_a,    img_b,    w_n,  m_n,  res_n,
         "No structure. Noise + dropout. Encoder is tempted to smooth."),
    ]
    for i, (title, A, B, W_img, mask, res, caption) in enumerate(rows, start=1):
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(A); ax.set_ylabel(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[i, 1]); ax.imshow(W_img); ax.set_title("W(A) -> B"); ax.axis("off")
        ax = fig.add_subplot(gs[i, 2]); ax.imshow(B); ax.set_title("Target B"); ax.axis("off")
        ax = fig.add_subplot(gs[i, 3]); ax.imshow(mask, cmap="gray", vmin=0, vmax=1); ax.set_title("Valid mask"); ax.axis("off")
        ax = fig.add_subplot(gs[i, 4]); im = ax.imshow(res, cmap="inferno", vmin=0, vmax=0.5)
        mean_err = res[mask].mean() if mask.any() else float("nan")
        ax.set_title(f"|W(A)-B|  (mean={mean_err:.3f})"); ax.axis("off")

    fig.suptitle("Alternatives to noisy inter-frame warps: how much structure does each preserve?", fontsize=14)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved: {out_path}")

    # Print a small numeric summary too.
    print("\nMean |W(A) - B| inside the valid mask (lower = better consistency target):")
    for title, _, _, _, mask, res, _ in rows:
        m = res[mask].mean() if mask.any() else float("nan")
        cov = mask.mean()
        print(f"  {title:38s}  err={m:.4f}   coverage={cov:.1%}")


if __name__ == "__main__":
    main()
