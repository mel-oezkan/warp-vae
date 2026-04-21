"""
Dataset factory for dynamic dataset instantiation.

Provides a unified interface for creating datasets based on configuration,
supporting multiple dataset types with optional Plucker coordinate loading.
"""

from typing import Dict, Any, Optional, Type
from torch.utils.data import Dataset

from ldm.util import instantiate_from_config


# Registry of available datasets
DATASET_REGISTRY: Dict[str, str] = {
    "co3d": "src.data.co3d_dataset.CO3DDataset",
    "omniobject": "src.data.omniobject3d_dataset.OmniObject3DDataset",
    "warp_co3d": "src.data.warp_dataset.WarpCO3DDataset",
    "precomputed_warp": "src.data.warp_dataset.PrecomputedWarpDataset",
    "imagenet": "src.data.imagenet_dataset.ImageNetDataset",
    # "mvimgnet": "src.data.mvimgnet_dataset.MVImgNetDataset",  # TODO: Implement if needed
}


def get_dataset(config: Dict[str, Any]) -> Dataset:
    """
    Factory function that returns appropriate dataset based on config.
    
    Supports two configuration styles:
    
    Style 1 - Direct target specification:
        data:
            dataset:
                target: src.data.co3d_dataset.CO3DDataset
                params:
                    root_dir: /path/to/data
                    include_plucker: true
    
    Style 2 - Dataset type shorthand:
        data:
            dataset:
                type: co3d
                params:
                    root_dir: /path/to/data
                    include_plucker: true
    
    Args:
        config: Dataset configuration dictionary with either:
                - 'target' and 'params' keys (instantiate_from_config style)
                - 'type' and 'params' keys (shorthand style)
    
    Returns:
        Instantiated Dataset object
        
    Raises:
        ValueError: If neither 'target' nor 'type' is specified
        KeyError: If 'type' is not in registry
    
    Example:
        >>> config = {
        ...     "target": "src.data.co3d_dataset.CO3DDataset",
        ...     "params": {"root_dir": "/data/co3d", "include_plucker": True}
        ... }
        >>> dataset = get_dataset(config)
    """
    # Style 1: Direct target specification
    if "target" in config:
        return instantiate_from_config(config)
    
    # Style 2: Dataset type shorthand
    if "type" in config:
        dataset_type = config["type"].lower()
        
        if dataset_type not in DATASET_REGISTRY:
            available = list(DATASET_REGISTRY.keys())
            raise KeyError(
                f"Unknown dataset type: '{dataset_type}'. "
                f"Available types: {available}"
            )
        
        # Build config for instantiate_from_config
        target = DATASET_REGISTRY[dataset_type]
        params = config.get("params", {})
        
        full_config = {
            "target": target,
            "params": params,
        }
        
        return instantiate_from_config(full_config)
    
    raise ValueError(
        "Dataset config must contain either 'target' or 'type' key. "
        f"Got keys: {list(config.keys())}"
    )


def register_dataset(name: str, target: str) -> None:
    """
    Register a new dataset type in the factory.
    
    Args:
        name: Short name for the dataset (e.g., "co3d")
        target: Full module path to dataset class
        
    Example:
        >>> register_dataset("custom", "my_module.CustomDataset")
    """
    DATASET_REGISTRY[name.lower()] = target
    print(f"[DatasetFactory] Registered dataset '{name}' -> {target}")


def list_available_datasets() -> Dict[str, str]:
    """
    List all registered dataset types.
    
    Returns:
        Dictionary mapping dataset names to their target classes
    """
    return DATASET_REGISTRY.copy()


def get_dataset_class(config: Dict[str, Any]) -> Type[Dataset]:
    """
    Get the dataset class without instantiating it.
    
    Useful for introspection or when you need to instantiate manually.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Dataset class (not an instance)
    """
    import importlib
    
    if "target" in config:
        target = config["target"]
    elif "type" in config:
        dataset_type = config["type"].lower()
        if dataset_type not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset type: '{dataset_type}'")
        target = DATASET_REGISTRY[dataset_type]
    else:
        raise ValueError("Config must contain 'target' or 'type'")
    
    # Parse module and class name
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)