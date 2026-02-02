"""Camera and multi-view geometry utilities."""

import json
from typing import List, Tuple
import numpy as np


def load_camera_data(transforms_path: str) -> dict:
    """Load camera transforms from JSON file."""
    with open(transforms_path) as f:
        data = json.load(f)
    return data


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


def find_overlapping_pairs(
    angular_sep: np.ndarray,
    max_angle: float = 30,
    min_angle: float = 5
) -> List[Tuple[int, int, float]]:
    """Find view pairs with angular separation in specified range.

    Args:
        angular_sep: (N, N) angular separation matrix in degrees
        max_angle: Maximum angular separation to consider as "overlapping"
        min_angle: Minimum angular separation (to avoid nearly identical views)

    Returns:
        List of (i, j, angle) tuples sorted by angle
    """
    n = angular_sep.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            angle = angular_sep[i, j]
            if min_angle <= angle <= max_angle:
                pairs.append((i, j, angle))
    return sorted(pairs, key=lambda x: x[2])


def find_view_sequences(
    positions: np.ndarray,
    angular_sep: np.ndarray,
    seq_length: int = 3,
    max_pairwise_angle: float = 30
) -> List[Tuple[Tuple[int, ...], float, float]]:
    """Find sequences of views that form a coherent sweep around the object.

    Uses greedy search starting from each view to find nearby views.

    Args:
        positions: (N, 3) camera positions
        angular_sep: (N, N) angular separation matrix in degrees
        seq_length: Number of views in each sequence
        max_pairwise_angle: Maximum angle between consecutive views in sequence

    Returns:
        List of tuples (view_indices, total_span_angle, avg_step_angle)
    """
    n = len(positions)
    sequences = []

    for start_idx in range(n):
        # Sort other views by angular distance from start
        distances = [(i, angular_sep[start_idx, i]) for i in range(n) if i != start_idx]
        distances.sort(key=lambda x: x[1])

        # Build sequence greedily
        sequence = [start_idx]
        current_idx = start_idx

        for _ in range(seq_length - 1):
            best_next = None
            best_angle = float('inf')

            for idx, _ in distances:
                if idx not in sequence:
                    angle = angular_sep[current_idx, idx]
                    if angle <= max_pairwise_angle and angle < best_angle:
                        best_next = idx
                        best_angle = angle

            if best_next is not None:
                sequence.append(best_next)
                current_idx = best_next
            else:
                break

        if len(sequence) == seq_length:
            total_span = angular_sep[sequence[0], sequence[-1]]
            step_angles = [angular_sep[sequence[i], sequence[i + 1]]
                          for i in range(len(sequence) - 1)]
            avg_step = np.mean(step_angles)
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
