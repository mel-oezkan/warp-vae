"""Evaluation script for OmniObject dataset.

This script provides visualization and verification tools for the OmniObject dataset:
1. Transformation matrices visualization
2. Camera positions 3D visualization
3. View pairs side-by-side with relative pose
4. Camera parameter verification
5. Plucker ray visualization
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms

from data_process.omniobject_dataset import OmniObjectDataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class OmniObjectEvaluator:
    """Evaluator for OmniObject dataset with multiple visualization methods."""

    def __init__(self, data_dir: str, output_dir: str = "./eval_results"):
        """Initialize evaluator.

        Args:
            data_dir: Path to OmniObject dataset root
            output_dir: Directory to save visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def visualize_transformation_matrices(self, obj_name: str, save=True):
        """Visualize all 24 transformation matrices for an object.

        Args:
            obj_name: Object name (e.g., "apple_001")
            save: Whether to save the plot
        """
        obj_dir = self.data_dir / "img" / obj_name
        transforms_file = obj_dir / "transforms.json"

        if not transforms_file.exists():
            print(f"Error: {transforms_file} not found")
            return

        with open(transforms_file) as f:
            camera_data = json.load(f)

        fig, axes = plt.subplots(6, 4, figsize=(20, 30))
        axes = axes.flatten()

        for idx, frame in enumerate(camera_data["frames"]):
            matrix = np.array(frame["transform_matrix"])

            ax = axes[idx]
            im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=-2, vmax=2)
            ax.set_title(f'Frame {idx:03d}', fontsize=12, weight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate matrix values
            for i in range(4):
                for j in range(4):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="w", fontsize=9)

        plt.suptitle(f'Transformation Matrices for {obj_name}', fontsize=16, weight='bold')
        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{obj_name}_transform_matrices.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved transformation matrices to {output_path}")

        plt.close()

    def visualize_camera_positions(self, obj_name: str, save=True):
        """3D plot of camera positions around the object.

        Args:
            obj_name: Object name (e.g., "apple_001")
            save: Whether to save the plot
        """
        obj_dir = self.data_dir / "img" / obj_name
        transforms_file = obj_dir / "transforms.json"

        if not transforms_file.exists():
            print(f"Error: {transforms_file} not found")
            return

        with open(transforms_file) as f:
            camera_data = json.load(f)

        # Extract camera centers (in C2W format, translation IS the camera center)
        positions = []
        for frame in camera_data["frames"]:
            matrix = np.array(frame["transform_matrix"])
            T_c2w = matrix[:3, 3]  # Camera center in world coordinates
            positions.append(T_c2w)

        positions = np.array(positions)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')

        # Color cameras by index
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=range(24), cmap='viridis', s=150, marker='o',
                           edgecolors='black', linewidths=1.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('View Index', fontsize=12)

        # Draw lines connecting sequential views
        for i in range(len(positions)):
            next_i = (i + 1) % len(positions)
            ax.plot([positions[i, 0], positions[next_i, 0]],
                   [positions[i, 1], positions[next_i, 1]],
                   [positions[i, 2], positions[next_i, 2]],
                   'k-', alpha=0.2, linewidth=1)

        # Plot origin (object center)
        ax.scatter([0], [0], [0], c='red', s=300, marker='*',
                  label='Object Center', edgecolors='black', linewidths=2)

        ax.set_xlabel('X', fontsize=12, weight='bold')
        ax.set_ylabel('Y', fontsize=12, weight='bold')
        ax.set_zlabel('Z', fontsize=12, weight='bold')
        ax.set_title(f'Camera Positions for {obj_name}', fontsize=14, weight='bold', pad=20)
        ax.legend(fontsize=11)

        # Set equal aspect ratio
        max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                             positions[:, 1].max()-positions[:, 1].min(),
                             positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save:
            output_path = self.output_dir / f"{obj_name}_camera_positions.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved camera positions to {output_path}")

        plt.close()

    def visualize_view_pairs(self, obj_name: str, num_pairs: int = 4, save=True):
        """Visualize view pairs side-by-side.

        Args:
            obj_name: Object name (e.g., "apple_001")
            num_pairs: Number of pairs to visualize
            save: Whether to save the plot
        """
        # Create dataset without normalization for visualization
        transform = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
        ])

        dataset = OmniObjectDataset(
            data_dir=str(self.data_dir),
            transform=transform,
            patch_num=8,
            image_size=512,
            pair_sampling="sequential"
        )

        # Find samples for this object
        obj_samples = [i for i, s in enumerate(dataset.samples)
                      if s["obj_dir"].name == obj_name]

        if len(obj_samples) == 0:
            print(f"No samples found for {obj_name}")
            return

        # Select pairs
        selected_indices = obj_samples[:min(num_pairs, len(obj_samples))]

        fig, axes = plt.subplots(len(selected_indices), 3, figsize=(18, 6*len(selected_indices)))
        if len(selected_indices) == 1:
            axes = axes.reshape(1, -1)

        for row, idx in enumerate(selected_indices):
            sample = dataset[idx]

            # View 1
            img1 = sample["image"].permute(1, 2, 0).numpy()
            axes[row, 0].imshow(img1)
            axes[row, 0].set_title(f'View {sample["view1_idx"]:03d}', fontsize=12, weight='bold')
            axes[row, 0].axis('off')

            # View 2
            img2 = sample["image2"].permute(1, 2, 0).numpy()
            axes[row, 1].imshow(img2)
            axes[row, 1].set_title(f'View {sample["view2_idx"]:03d}', fontsize=12, weight='bold')
            axes[row, 1].axis('off')

            # Relative pose info
            R_rel = sample["R_rel"].numpy()
            T_rel = sample["T_rel"].numpy()

            axes[row, 2].text(0.05, 0.95, "Relative Rotation:",
                            transform=axes[row, 2].transAxes, fontsize=11, weight='bold',
                            verticalalignment='top')

            # Format rotation matrix nicely
            R_str = "[\n"
            for i in range(3):
                R_str += "  [" + ", ".join([f"{R_rel[i,j]:7.4f}" for j in range(3)]) + "]\n"
            R_str += "]"

            axes[row, 2].text(0.05, 0.85, R_str,
                            transform=axes[row, 2].transAxes, fontsize=9, family='monospace',
                            verticalalignment='top')

            axes[row, 2].text(0.05, 0.45, "Relative Translation:",
                            transform=axes[row, 2].transAxes, fontsize=11, weight='bold',
                            verticalalignment='top')

            T_str = f"[{T_rel[0]:7.4f}, {T_rel[1]:7.4f}, {T_rel[2]:7.4f}]"
            axes[row, 2].text(0.05, 0.35, T_str,
                            transform=axes[row, 2].transAxes, fontsize=9, family='monospace',
                            verticalalignment='top')

            # Translation magnitude
            T_norm = np.linalg.norm(T_rel)
            axes[row, 2].text(0.05, 0.25, f"||T_rel|| = {T_norm:.4f}",
                            transform=axes[row, 2].transAxes, fontsize=10,
                            verticalalignment='top')

            axes[row, 2].axis('off')

        plt.suptitle(f'View Pairs for {obj_name}', fontsize=16, weight='bold')
        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{obj_name}_view_pairs.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved view pairs to {output_path}")

        plt.close()

    def verify_camera_parameters(self, obj_name: str, save=True):
        """Verify camera parameter extraction is correct.

        Args:
            obj_name: Object name (e.g., "apple_001")
            save: Whether to save the plot
        """
        obj_dir = self.data_dir / "img" / obj_name
        transforms_file = obj_dir / "transforms.json"

        if not transforms_file.exists():
            print(f"Error: {transforms_file} not found")
            return

        with open(transforms_file) as f:
            camera_data = json.load(f)

        # Create dataset to get extracted params
        dataset = OmniObjectDataset(
            data_dir=str(self.data_dir),
            transform=None,
            patch_num=8,
            image_size=512,
        )

        # Check consistency
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Focal lengths
        focal_lengths = []
        for frame in camera_data["frames"]:
            params = dataset._extract_camera_params(frame, camera_data["camera_angle_x"], 512)
            focal_lengths.append(params["focal_length"][0].item())

        axes[0, 0].plot(focal_lengths, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].axhline(y=np.mean(focal_lengths), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(focal_lengths):.2f}')
        axes[0, 0].set_title('Focal Length Consistency', fontsize=13, weight='bold')
        axes[0, 0].set_xlabel('Frame Index', fontsize=11)
        axes[0, 0].set_ylabel('Focal Length (pixels)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)

        # 2. Rotation matrix determinants (should be 1)
        determinants = []
        for frame in camera_data["frames"]:
            params = dataset._extract_camera_params(frame, camera_data["camera_angle_x"], 512)
            det = torch.det(params["R"]).item()
            determinants.append(det)

        axes[0, 1].plot(determinants, 'o-', linewidth=2, markersize=8, color='forestgreen')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected: 1.0')
        axes[0, 1].set_title('Rotation Matrix Determinant', fontsize=13, weight='bold')
        axes[0, 1].set_xlabel('Frame Index', fontsize=11)
        axes[0, 1].set_ylabel('Determinant', fontsize=11)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Add text with statistics
        det_mean = np.mean(determinants)
        det_std = np.std(determinants)
        axes[0, 1].text(0.02, 0.98, f'Mean: {det_mean:.6f}\nStd: {det_std:.6f}',
                       transform=axes[0, 1].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. Translation magnitudes
        translation_norms = []
        for frame in camera_data["frames"]:
            params = dataset._extract_camera_params(frame, camera_data["camera_angle_x"], 512)
            norm = torch.norm(params["T"]).item()
            translation_norms.append(norm)

        axes[1, 0].plot(translation_norms, 'o-', linewidth=2, markersize=8, color='darkorange')
        axes[1, 0].axhline(y=np.mean(translation_norms), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(translation_norms):.2f}')
        axes[1, 0].set_title('Translation Vector Magnitude', fontsize=13, weight='bold')
        axes[1, 0].set_xlabel('Frame Index', fontsize=11)
        axes[1, 0].set_ylabel('||T||', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=10)

        # 4. Orthogonality check: R @ R^T should be identity
        ortho_errors = []
        for frame in camera_data["frames"]:
            params = dataset._extract_camera_params(frame, camera_data["camera_angle_x"], 512)
            R = params["R"]
            identity_error = torch.norm(R @ R.T - torch.eye(3)).item()
            ortho_errors.append(identity_error)

        axes[1, 1].plot(ortho_errors, 'o-', linewidth=2, markersize=8, color='darkviolet')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Expected: 0')
        axes[1, 1].set_title('Rotation Orthogonality Error', fontsize=13, weight='bold')
        axes[1, 1].set_xlabel('Frame Index', fontsize=11)
        axes[1, 1].set_ylabel('||R @ R^T - I||', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=10)

        # Add text with max error
        max_error = np.max(ortho_errors)
        axes[1, 1].text(0.02, 0.98, f'Max error: {max_error:.2e}',
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Camera Parameter Verification for {obj_name}', fontsize=16, weight='bold')
        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{obj_name}_camera_verification.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved camera verification to {output_path}")

        plt.close()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Camera Parameter Verification Summary for {obj_name}")
        print(f"{'='*60}")
        print(f"Focal length consistency: {np.std(focal_lengths):.6f} (std dev)")
        print(f"Rotation determinant: {det_mean:.6f} ± {det_std:.6f} (should be 1.0)")
        print(f"Orthogonality error: max {max_error:.2e} (should be ~0)")
        print(f"Translation magnitude: {np.mean(translation_norms):.4f} ± {np.std(translation_norms):.4f}")
        print(f"{'='*60}\n")

    def visualize_plucker_rays(self, obj_name: str, view_idx: int = 0, save=True):
        """Visualize Plucker rays for a single view.

        Args:
            obj_name: Object name (e.g., "apple_001")
            view_idx: View index to visualize
            save: Whether to save the plot
        """
        # Create dataset
        transform = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
        ])

        dataset = OmniObjectDataset(
            data_dir=str(self.data_dir),
            transform=transform,
            patch_num=8,  # 8x8 grid
            image_size=512,
        )

        # Find a sample for this object and view
        sample_idx = None
        for i, sample in enumerate(dataset.samples):
            if sample["obj_dir"].name == obj_name and sample["view1_idx"] == view_idx:
                sample_idx = i
                break

        if sample_idx is None:
            print(f"No sample found for {obj_name} view {view_idx}")
            return

        sample_data = dataset[sample_idx]
        pluck_rays = sample_data["pluck_ray"]  # Shape: (64, 6) for 8x8 grid

        # Extract directions and moments
        directions = pluck_rays[:, :3].numpy()
        moments = pluck_rays[:, 3:].numpy()

        # Compute origins: O = D x M (cross product)
        origins = np.cross(directions, moments, axis=1)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')

        # Plot rays as arrows
        for i in range(len(origins)):
            origin = origins[i]
            direction = directions[i]

            # Plot ray as arrow
            ax.quiver(origin[0], origin[1], origin[2],
                     direction[0], direction[1], direction[2],
                     length=0.3, alpha=0.7, arrow_length_ratio=0.2,
                     color=plt.cm.viridis(i / len(origins)))

        # Plot a small sphere at origin to represent rays converging
        ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2],
                  c=range(len(origins)), cmap='viridis', s=30, alpha=0.5)

        ax.set_xlabel('X', fontsize=12, weight='bold')
        ax.set_ylabel('Y', fontsize=12, weight='bold')
        ax.set_zlabel('Z', fontsize=12, weight='bold')
        ax.set_title(f'Plucker Rays for {obj_name} - View {view_idx:03d}\n(8x8 grid = 64 rays)',
                    fontsize=14, weight='bold', pad=20)

        if save:
            output_path = self.output_dir / f"{obj_name}_view{view_idx:03d}_plucker_rays.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved Plucker rays to {output_path}")

        plt.close()

    def run_full_evaluation(self, obj_name: str):
        """Run all evaluation visualizations for an object.

        Args:
            obj_name: Object name (e.g., "apple_001")
        """
        print(f"\n{'='*70}")
        print(f"Running Full Evaluation for: {obj_name}")
        print(f"{'='*70}\n")

        print("1. Visualizing transformation matrices...")
        self.visualize_transformation_matrices(obj_name)

        print("2. Visualizing camera positions...")
        self.visualize_camera_positions(obj_name)

        print("3. Visualizing view pairs...")
        self.visualize_view_pairs(obj_name, num_pairs=4)

        print("4. Verifying camera parameters...")
        self.verify_camera_parameters(obj_name)

        print("5. Visualizing Plucker rays...")
        self.visualize_plucker_rays(obj_name, view_idx=0)

        print(f"\n{'='*70}")
        print(f"Evaluation Complete! Results saved to: {self.output_dir}/")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OmniObject dataset with multiple visualization methods"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
        help="Path to OmniObject dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for visualization plots"
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default="apple_001",
        help="Object name to evaluate (e.g., apple_001, plant_014)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "transforms", "positions", "pairs", "verify", "plucker"],
        help="Which evaluation method to run"
    )

    args = parser.parse_args()

    evaluator = OmniObjectEvaluator(args.data_dir, args.output_dir)

    if args.method == "all":
        evaluator.run_full_evaluation(args.object_name)
    elif args.method == "transforms":
        evaluator.visualize_transformation_matrices(args.object_name)
    elif args.method == "positions":
        evaluator.visualize_camera_positions(args.object_name)
    elif args.method == "pairs":
        evaluator.visualize_view_pairs(args.object_name)
    elif args.method == "verify":
        evaluator.verify_camera_parameters(args.object_name)
    elif args.method == "plucker":
        evaluator.visualize_plucker_rays(args.object_name)


if __name__ == "__main__":
    main()
