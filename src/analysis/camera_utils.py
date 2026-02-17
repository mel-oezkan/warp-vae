"""Camera and multi-view geometry utilities."""

import gzip
import json
from typing import List, Tuple
import numpy as np


def load_camera_data(transforms_path: str) -> dict:
    """Load camera transforms from JSON file."""
    with open(transforms_path) as f:
        data = json.load(f)
    return data


def load_co3d_annotations(annotation_path: str) -> dict:
    """Load CO3D preprocessed annotations from .jgz file.

    Args:
        annotation_path: Path to a preprocessed .jgz annotation file
            (e.g. hydrant_train.jgz produced by preprocess_co3d.py)

    Returns:
        Dictionary mapping sequence_name -> list of frame dicts.
        Each frame dict has: filepath, R, T, focal_length, principal_point, bbox
    """
    with gzip.open(annotation_path, "r") as f:
        data = json.loads(f.read())
    return data


def extract_co3d_camera_positions(frames: list) -> np.ndarray:
    """Extract camera world positions from CO3D W2C (R, T) format.

    CO3D convention: R is 3x3 world-to-camera rotation, T is 3D W2C translation.
    Camera world position = -R^T @ T

    Args:
        frames: List of frame dicts with 'R' and 'T' keys

    Returns:
        Array of shape (N, 3) with camera world positions
    """
    positions = []
    for frame in frames:
        R = np.array(frame["R"])
        T = np.array(frame["T"])
        pos = -R.T @ T
        positions.append(pos)
    return np.array(positions)


def extract_camera_positions(camera_data: dict) -> np.ndarray:
    """Extract camera positions from transform matrices.

    Args:
        camera_data: Dictionary with 'frames' containing transform matrices

    Returns:
        Array of shape (N, 3) with camera positions
    """
    positions = []
    for frame in camera_data["frames"]:
        transform = np.array(frame["transform_matrix"])
        position = transform[:3, 3]
        positions.append(position)
    return np.array(positions)


def compute_angular_separation(positions: np.ndarray) -> np.ndarray:
    """Compute angular separation matrix between all camera positions.

    Assumes cameras are looking at the origin, so angular separation
    is computed as the angle between position vectors.

    Args:
        positions: Array of shape (N, 3) with camera positions

    Returns:
        Angular separation matrix of shape (N, N) in degrees
    """
    n_views = len(positions)
    angular_sep = np.zeros((n_views, n_views))

    for i in range(n_views):
        for j in range(n_views):
            if i == j:
                angular_sep[i, j] = 0
            else:
                dot = np.dot(positions[i], positions[j])
                norm_prod = np.linalg.norm(positions[i]) * np.linalg.norm(positions[j])
                cos_angle = np.clip(dot / (norm_prod + 1e-8), -1, 1)
                angular_sep[i, j] = np.arccos(cos_angle) * 180 / np.pi

    return angular_sep


def compute_euclidean_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between camera positions.

    Works with any dataset (OmniObject3D or CO3D) since it operates
    directly on camera world positions.

    Args:
        positions: Array of shape (N, 3) with camera world positions

    Returns:
        Distance matrix of shape (N, N)
    """
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def compute_camera_distance_matrix(frames: list) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between camera positions.

    Unlike angular separation, this captures both rotational and
    translational differences (e.g. camera moving closer/farther from object).

    Args:
        frames: List of frame dicts with 'R' and 'T' keys

    Returns:
        Distance matrix of shape (N, N)
    """
    positions = extract_co3d_camera_positions(frames)
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def compute_relative_pose_distance(
    frames: list,
    weight_rotation: float = 1.0,
    weight_translation: float = 1.0,
) -> np.ndarray:
    """Compute pairwise pose distance combining rotation and translation.

    Rotation distance is the geodesic angle (degrees) between R_i and R_j.
    Translation distance is Euclidean distance between camera centers.
    Both are normalized to [0, 1] before weighting.

    Args:
        frames: List of frame dicts with 'R' and 'T' keys
        weight_rotation: Weight for rotation component
        weight_translation: Weight for translation component

    Returns:
        Combined distance matrix of shape (N, N)
    """
    n = len(frames)
    Rs = [np.array(f["R"]) for f in frames]
    positions = extract_co3d_camera_positions(frames)

    rot_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Relative rotation: R_rel = R_i @ R_j^T
            R_rel = Rs[i] @ Rs[j].T
            # Geodesic angle: arccos((trace(R_rel) - 1) / 2)
            trace = np.clip(np.trace(R_rel), -1.0, 3.0)
            angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            rot_dist[i, j] = rot_dist[j, i] = np.degrees(angle)

    diff = positions[:, None, :] - positions[None, :, :]
    trans_dist = np.linalg.norm(diff, axis=-1)

    # Normalize each to [0, 1]
    rot_norm = rot_dist / (rot_dist.max() + 1e-8)
    trans_norm = trans_dist / (trans_dist.max() + 1e-8)

    return weight_rotation * rot_norm + weight_translation * trans_norm


def find_overlapping_pairs(
    distance_matrix: np.ndarray,
    max_distance: float = 3.0,
    min_distance: float = 0.5
) -> List[Tuple[int, int, float]]:
    """Find view pairs with distance in specified range.

    Works with any pairwise distance matrix (Euclidean camera distance).

    Args:
        distance_matrix: (N, N) pairwise distance matrix
        max_distance: Maximum distance to consider
        min_distance: Minimum distance (to avoid nearly identical views)

    Returns:
        List of (i, j, distance) tuples sorted by distance
    """
    n = distance_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_matrix[i, j]
            if min_distance <= dist <= max_distance:
                pairs.append((i, j, dist))
    return sorted(pairs, key=lambda x: x[2])


def find_view_sequences(
    positions: np.ndarray,
    dist_matrix: np.ndarray,
    seq_length: int = 3,
    max_pairwise_angle: float = 3.0
) -> List[Tuple[Tuple[int, ...], float, float]]:
    """Find sequences of views that form a coherent sweep around the object.

    Uses greedy search starting from each view to find nearby views.

    Args:
        positions: (N, 3) camera positions
        dist_matrix: (N, N) pairwise distance matrix
        seq_length: Number of views in each sequence
        max_pairwise_angle: Maximum distance between consecutive views

    Returns:
        List of tuples (view_indices, total_span, avg_step)
    """
    n = len(positions)
    sequences = []

    for start_idx in range(n):
        # Sort other views by distance from start
        distances = [(i, dist_matrix[start_idx, i]) for i in range(n) if i != start_idx]
        distances.sort(key=lambda x: x[1])

        # Build sequence greedily
        sequence = [start_idx]
        current_idx = start_idx

        for _ in range(seq_length - 1):
            best_next = None
            best_dist = float('inf')

            for idx, _ in distances:
                if idx not in sequence:
                    d = dist_matrix[current_idx, idx]
                    if d <= max_pairwise_angle and d < best_dist:
                        best_next = idx
                        best_dist = d

            if best_next is not None:
                sequence.append(best_next)
                current_idx = best_next
            else:
                break

        if len(sequence) == seq_length:
            total_span = dist_matrix[sequence[0], sequence[-1]]
            step_dists = [dist_matrix[sequence[i], sequence[i + 1]]
                         for i in range(len(sequence) - 1)]
            avg_step = np.mean(step_dists)
            sequences.append((tuple(sequence), total_span, avg_step))

    # Remove duplicate sequences (same views, different order)
    unique_sequences = []
    seen_sets = set()
    for seq, span, avg in sequences:
        seq_set = frozenset(seq)
        if seq_set not in seen_sets:
            seen_sets.add(seq_set)
            unique_sequences.append((seq, span, avg))

    return unique_sequences
