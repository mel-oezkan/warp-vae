"""
Memory profiler utility for tracking GPU memory usage during VAE training.

Provides tools to snapshot, compare, and profile GPU memory consumption.
"""

import torch
from typing import Dict, Optional
import functools


class MemoryProfiler:
    """Utility for tracking GPU memory usage during training."""

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory profiler.

        Args:
            device: CUDA device to monitor (defaults to cuda:0)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snapshots = {}

    def snapshot(self, name: str) -> Dict[str, float]:
        """
        Take a memory snapshot with given name.

        Args:
            name: Identifier for this snapshot

        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2

        snapshot = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
        }
        self.snapshots[name] = snapshot
        return snapshot

    def compare(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Compare two snapshots.

        Args:
            name1: First snapshot identifier
            name2: Second snapshot identifier

        Returns:
            Dictionary with memory differences in MB
        """
        if name1 not in self.snapshots or name2 not in self.snapshots:
            return {}

        s1, s2 = self.snapshots[name1], self.snapshots[name2]
        return {
            'allocated_diff_mb': s2['allocated_mb'] - s1['allocated_mb'],
            'reserved_diff_mb': s2['reserved_mb'] - s1['reserved_mb'],
        }

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_summary(self) -> str:
        """
        Get formatted summary of current memory usage.

        Returns:
            Human-readable memory summary string
        """
        if not torch.cuda.is_available():
            return "CUDA not available"

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2

        return (
            f"GPU Memory: "
            f"Allocated={allocated:.1f}MB, "
            f"Reserved={reserved:.1f}MB, "
            f"Peak={max_allocated:.1f}MB"
        )

    @staticmethod
    def profile_function(func):
        """
        Decorator to profile memory usage of a function.

        Args:
            func: Function to profile

        Returns:
            Wrapped function that prints memory statistics
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated() / 1024**2

            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated() / 1024**2
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"[MemProfile] {func.__name__}: "
                      f"Δ={end_mem - start_mem:.1f}MB, Peak={peak_mem:.1f}MB")

            return result
        return wrapper
