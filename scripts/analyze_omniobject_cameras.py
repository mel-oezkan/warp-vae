"""Analyze OmniObject3D camera positions and visualize overlap potential."""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_camera_data(transforms_path):
    """Load camera transforms from JSON file."""
    with open(transforms_path) as f:
        data = json.load(f)
    return data


def extract_camera_positions(camera_data):
    """Extract camera positions and viewing directions from transform matrices."""
    positions = []
    directions = []

    for frame in camera_data["frames"]:
        # Transform matrix is camera-to-world (C2W)
        transform = np.array(frame["transform_matrix"])

        # Camera position is the translation column (last column)
        position = transform[:3, 3]
        positions.append(position)

        # Camera viewing direction is -Z axis in camera space transformed to world
        # The third column of rotation gives the camera's Z axis in world coords
        # Camera looks along -Z, so viewing direction is -R[:,2]
        direction = -transform[:3, 2]
        directions.append(direction)

    return np.array(positions), np.array(directions)


def compute_view_overlap_matrix(positions, directions, fov_rad):
    """Compute overlap potential between view pairs based on viewing frustums.

    Args:
        positions: (N, 3) camera positions
        directions: (N, 3) viewing directions (normalized)
        fov_rad: Field of view in radians

    Returns:
        overlap_matrix: (N, N) matrix where entry (i,j) is the overlap score
    """
    n_views = len(positions)
    overlap_matrix = np.zeros((n_views, n_views))

    # Normalize directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    half_fov = fov_rad / 2

    for i in range(n_views):
        for j in range(n_views):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue

            # Compute angle between viewing directions
            dot_product = np.dot(directions[i], directions[j])
            dot_product = np.clip(dot_product, -1, 1)
            angle_between = np.arccos(dot_product)

            # Views pointing in similar directions have higher overlap potential
            # Views pointing at each other (angle ~ 180°) might see same object from opposite sides

            # Method 1: Direct angle similarity (views looking same direction)
            direction_similarity = np.cos(angle_between)

            # Method 2: Both cameras point toward object center (assumed at origin)
            # Vector from camera to origin
            to_center_i = -positions[i] / (np.linalg.norm(positions[i]) + 1e-8)
            to_center_j = -positions[j] / (np.linalg.norm(positions[j]) + 1e-8)

            # Check if both cameras are looking at roughly the same part of the object
            # by checking angular distance between camera positions as seen from origin
            pos_angle = np.arccos(np.clip(
                np.dot(positions[i], positions[j]) /
                (np.linalg.norm(positions[i]) * np.linalg.norm(positions[j]) + 1e-8),
                -1, 1
            ))

            # Overlap score: higher when cameras are close together (similar viewpoints)
            # and when they're looking at the object
            overlap_matrix[i, j] = np.exp(-pos_angle / (np.pi / 4))  # Decay based on angular separation

    return overlap_matrix


def analyze_view_distribution(positions):
    """Analyze the spatial distribution of camera positions."""
    # Compute distances from origin
    distances = np.linalg.norm(positions, axis=1)

    # Compute pairwise distances between cameras
    n = len(positions)
    pairwise_distances = []
    for i in range(n):
        for j in range(i+1, n):
            pairwise_distances.append(np.linalg.norm(positions[i] - positions[j]))
    pairwise_distances = np.array(pairwise_distances)

    # Compute angular coverage
    angles_xy = np.arctan2(positions[:, 1], positions[:, 0])
    angles_elevation = np.arcsin(positions[:, 2] / (np.linalg.norm(positions, axis=1) + 1e-8))

    return {
        "distance_to_origin": {
            "mean": np.mean(distances),
            "std": np.std(distances),
            "min": np.min(distances),
            "max": np.max(distances),
        },
        "pairwise_distances": {
            "mean": np.mean(pairwise_distances),
            "std": np.std(pairwise_distances),
            "min": np.min(pairwise_distances),
            "max": np.max(pairwise_distances),
        },
        "azimuth_angles_deg": {
            "range": (np.min(angles_xy) * 180/np.pi, np.max(angles_xy) * 180/np.pi),
            "coverage": (np.max(angles_xy) - np.min(angles_xy)) * 180/np.pi,
        },
        "elevation_angles_deg": {
            "range": (np.min(angles_elevation) * 180/np.pi, np.max(angles_elevation) * 180/np.pi),
            "coverage": (np.max(angles_elevation) - np.min(angles_elevation)) * 180/np.pi,
        }
    }


def visualize_cameras(positions, directions, fov_rad, overlap_matrix, output_path):
    """Create 3D visualization of camera positions and orientations."""
    fig = plt.figure(figsize=(20, 15))

    # Main 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot camera positions
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c=np.arange(len(positions)), cmap='viridis', s=100, label='Camera positions')

    # Plot viewing directions as arrows
    arrow_scale = 0.3
    for i, (pos, dir) in enumerate(zip(positions, directions)):
        ax1.quiver(pos[0], pos[1], pos[2],
                   dir[0]*arrow_scale, dir[1]*arrow_scale, dir[2]*arrow_scale,
                   color='red', alpha=0.6, arrow_length_ratio=0.3)

    # Plot object center
    ax1.scatter([0], [0], [0], c='black', s=200, marker='*', label='Object center')

    # Draw simplified view frustums (just showing FOV cone)
    for i, (pos, dir) in enumerate(zip(positions, directions)):
        # Normalize direction
        dir_norm = dir / np.linalg.norm(dir)

        # Draw line from camera to object center
        ax1.plot([pos[0], 0], [pos[1], 0], [pos[2], 0],
                 'g--', alpha=0.2, linewidth=0.5)

    # Add camera indices
    for i, pos in enumerate(positions):
        ax1.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, str(i), fontsize=8)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Camera Positions and Viewing Directions')
    ax1.legend()

    # Set equal aspect ratio
    max_range = np.max(np.abs(positions)) * 1.2
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(positions[:, 0], positions[:, 1], c=np.arange(len(positions)),
                cmap='viridis', s=100)
    ax2.scatter([0], [0], c='black', s=200, marker='*')
    for i, pos in enumerate(positions):
        ax2.annotate(str(i), (pos[0], pos[1]), fontsize=8)
        ax2.arrow(pos[0], pos[1], directions[i, 0]*0.2, directions[i, 1]*0.2,
                  head_width=0.05, head_length=0.02, fc='red', ec='red', alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-Down View (XY plane)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Overlap heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(overlap_matrix, cmap='YlOrRd', aspect='equal')
    ax3.set_xlabel('View Index')
    ax3.set_ylabel('View Index')
    ax3.set_title('View Pair Overlap Potential')
    plt.colorbar(im, ax=ax3, label='Overlap Score')

    # Elevation vs Azimuth
    ax4 = fig.add_subplot(2, 2, 4)
    azimuth = np.arctan2(positions[:, 1], positions[:, 0]) * 180 / np.pi
    elevation = np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1)) * 180 / np.pi
    scatter = ax4.scatter(azimuth, elevation, c=np.arange(len(positions)),
                          cmap='viridis', s=100)
    for i, (az, el) in enumerate(zip(azimuth, elevation)):
        ax4.annotate(str(i), (az, el), fontsize=8)
    ax4.set_xlabel('Azimuth (degrees)')
    ax4.set_ylabel('Elevation (degrees)')
    ax4.set_title('Camera Angular Distribution')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='View Index')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def find_overlapping_pairs(overlap_matrix, threshold=0.5):
    """Find pairs of views with high overlap potential."""
    n = overlap_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if overlap_matrix[i, j] >= threshold:
                pairs.append((i, j, overlap_matrix[i, j]))
    return sorted(pairs, key=lambda x: -x[2])


def main():
    # Load sample object
    data_dir = Path("/data/lab_moezkan/omni_obj/blender_renders_24_views/img")
    sample_obj = data_dir / "anise_001"
    transforms_path = sample_obj / "transforms.json"

    print("=" * 60)
    print("OmniObject3D Camera Analysis")
    print("=" * 60)

    # Load camera data
    camera_data = load_camera_data(transforms_path)
    fov_rad = camera_data["camera_angle_x"]
    print(f"\nField of View: {fov_rad:.4f} rad ({np.degrees(fov_rad):.2f}°)")
    print(f"Number of views: {len(camera_data['frames'])}")

    # Extract camera positions and directions
    positions, directions = extract_camera_positions(camera_data)

    # Analyze view distribution
    print("\n" + "-" * 40)
    print("Camera Distribution Analysis:")
    print("-" * 40)
    stats = analyze_view_distribution(positions)

    print(f"\nDistance to origin:")
    print(f"  Mean: {stats['distance_to_origin']['mean']:.3f}")
    print(f"  Std:  {stats['distance_to_origin']['std']:.3f}")
    print(f"  Range: [{stats['distance_to_origin']['min']:.3f}, {stats['distance_to_origin']['max']:.3f}]")

    print(f"\nPairwise camera distances:")
    print(f"  Mean: {stats['pairwise_distances']['mean']:.3f}")
    print(f"  Std:  {stats['pairwise_distances']['std']:.3f}")
    print(f"  Range: [{stats['pairwise_distances']['min']:.3f}, {stats['pairwise_distances']['max']:.3f}]")

    print(f"\nAzimuth coverage:")
    print(f"  Range: [{stats['azimuth_angles_deg']['range'][0]:.1f}°, {stats['azimuth_angles_deg']['range'][1]:.1f}°]")
    print(f"  Coverage: {stats['azimuth_angles_deg']['coverage']:.1f}°")

    print(f"\nElevation coverage:")
    print(f"  Range: [{stats['elevation_angles_deg']['range'][0]:.1f}°, {stats['elevation_angles_deg']['range'][1]:.1f}°]")
    print(f"  Coverage: {stats['elevation_angles_deg']['coverage']:.1f}°")

    # Compute overlap matrix
    overlap_matrix = compute_view_overlap_matrix(positions, directions, fov_rad)

    # Find highly overlapping pairs
    print("\n" + "-" * 40)
    print("View Pair Overlap Analysis:")
    print("-" * 40)

    overlapping_pairs = find_overlapping_pairs(overlap_matrix, threshold=0.6)
    print(f"\nPairs with overlap score >= 0.6: {len(overlapping_pairs)}")
    print("\nTop 10 overlapping pairs:")
    for i, j, score in overlapping_pairs[:10]:
        angular_sep = np.arccos(np.clip(
            np.dot(positions[i], positions[j]) /
            (np.linalg.norm(positions[i]) * np.linalg.norm(positions[j])),
            -1, 1
        )) * 180 / np.pi
        print(f"  Views {i:2d} & {j:2d}: score={score:.3f}, angular separation={angular_sep:.1f}°")

    # Statistics for multi-view evaluation
    print("\n" + "-" * 40)
    print("Multi-View Evaluation Suitability:")
    print("-" * 40)

    # Compute consecutive pair statistics
    consecutive_overlaps = []
    for i in range(len(positions)):
        j = (i + 1) % len(positions)
        consecutive_overlaps.append(overlap_matrix[i, j])

    print(f"\nConsecutive view pairs (0-1, 1-2, ..., 23-0):")
    print(f"  Mean overlap score: {np.mean(consecutive_overlaps):.3f}")
    print(f"  Min overlap score:  {np.min(consecutive_overlaps):.3f}")
    print(f"  Max overlap score:  {np.max(consecutive_overlaps):.3f}")

    # Check if any pairs have very high overlap (good for MVS)
    high_overlap_count = np.sum(overlap_matrix > 0.7) - len(positions)  # Exclude diagonal
    total_pairs = len(positions) * (len(positions) - 1)
    print(f"\nPairs with high overlap (>0.7): {high_overlap_count//2} / {total_pairs//2} ({100*high_overlap_count/total_pairs:.1f}%)")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    if stats['azimuth_angles_deg']['coverage'] > 300:
        print("✓ Good azimuth coverage (>300°) - cameras surround the object")
    else:
        print("⚠ Limited azimuth coverage - views might miss some object sides")

    if stats['elevation_angles_deg']['coverage'] > 30:
        print("✓ Good elevation variation - multiple viewing angles")
    else:
        print("⚠ Limited elevation variation - mostly horizontal views")

    if np.mean(consecutive_overlaps) > 0.5:
        print("✓ Good consecutive overlap - suitable for sequential multi-view tasks")
    else:
        print("⚠ Low consecutive overlap - may need different pairing strategy")

    # Save visualization
    output_path = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/omniobj_camera_visualization.png")
    visualize_cameras(positions, directions, fov_rad, overlap_matrix, output_path)

    return positions, directions, overlap_matrix, stats


if __name__ == "__main__":
    main()
