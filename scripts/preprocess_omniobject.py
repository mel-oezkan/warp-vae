"""
Convert OmniObject3D transforms.json files to the .jgz annotation format
expected by precompute_warps.py and PrecomputedWarpDataset.

Each object directory becomes a "sequence" with 24 frames. Camera parameters
are converted from C2W (Blender/NeRF convention) to W2C (CO3D convention).

Usage:
    python preprocess_omniobject.py \
        --data_root /data/lab_moezkan/omni_obj/blender_renders_24_views \
        --output_file data/omniobject_annotations/omniobject_all.jgz

    # Small subset for testing:
    python preprocess_omniobject.py \
        --data_root /data/lab_moezkan/omni_obj/blender_renders_24_views \
        --output_file data/omniobject_annotations/omniobject_100.jgz \
        --max_objects 100
"""

import argparse
import gzip
import json
import os
from math import tan
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Convert OmniObject3D to .jgz annotation format"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing img/ subdirectory"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Output .jgz annotation file path"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Reference image size for focal length computation (default: 256)"
    )
    parser.add_argument(
        "--max_objects", type=int, default=None,
        help="Limit to first N objects (for testing)"
    )
    args = parser.parse_args()

    img_dir = Path(args.data_root) / "img"
    if not img_dir.exists():
        print(f"ERROR: {img_dir} does not exist")
        return

    objects = sorted([d.name for d in img_dir.iterdir() if d.is_dir()])
    if args.max_objects:
        objects = objects[:args.max_objects]

    print(f"Processing {len(objects)} objects from {img_dir}")

    annotations = {}
    skipped = 0

    for obj_name in tqdm(objects, desc="Processing objects"):
        transforms_path = img_dir / obj_name / "transforms.json"
        if not transforms_path.exists():
            skipped += 1
            continue

        with open(transforms_path) as f:
            camera_data = json.load(f)

        camera_angle_x = camera_data["camera_angle_x"]
        focal = (args.image_size / 2) / tan(camera_angle_x / 2)

        frames = []
        for i, frame in enumerate(camera_data["frames"]):
            M = np.array(frame["transform_matrix"])
            R_c2w = M[:3, :3]
            T_c2w = M[:3, 3]

            # Convert to W2C (CO3D convention)
            R_w2c = R_c2w.T
            T_w2c = -R_c2w.T @ T_c2w

            frames.append({
                "filepath": f"img/{obj_name}/{i:03d}.png",
                "R": R_w2c.tolist(),
                "T": T_w2c.tolist(),
                "focal_length": [float(focal), float(focal)],
                "principal_point": [args.image_size / 2, args.image_size / 2],
            })

        annotations[obj_name] = frames

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with gzip.open(args.output_file, "w") as f:
        f.write(json.dumps(annotations).encode("utf-8"))

    total_frames = sum(len(v) for v in annotations.values())
    print(f"Saved {len(annotations)} objects, {total_frames} total frames to {args.output_file}")
    if skipped:
        print(f"Skipped {skipped} objects (missing transforms.json)")


if __name__ == "__main__":
    main()
