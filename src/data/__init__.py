"""
Data module for VAE training.

Provides:
- Dataset factory for dynamic dataset instantiation
- Unified DataModule for all dataset types
- Individual dataset implementations
"""

from src.data.dataset_factory import (
    get_dataset,
    register_dataset,
    list_available_datasets,
    get_dataset_class,
    DATASET_REGISTRY,
)

from src.data.datamodule import (
    VAEDataModule,
    PairedVAEDataModule,
)

# Import dataset classes for registration
# These will be imported when the module is loaded
try:
    from src.data.co3d_dataset import CO3DDataset
except ImportError:
    pass

try:
    from src.data.omniobject3d_dataset import OmniObject3DDataset
except ImportError:
    pass

try:
    from src.data.mvimgnet_dataset import MVImgNetDataset
except ImportError:
    pass


__all__ = [
    # Factory functions
    "get_dataset",
    "register_dataset", 
    "list_available_datasets",
    "get_dataset_class",
    "DATASET_REGISTRY",

    # DataModules
    "VAEDataModule",
    "PairedVAEDataModule",
    
    # Datasets
    "CO3DDataset",
    "OmniObject3DDataset",
    "MVImgNetDataset",
]