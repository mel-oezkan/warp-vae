"""
Unified DataModule for VAE training.

Provides a single DataModule class that can handle any dataset type
through configuration-based instantiation.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Dict, Any, Optional, Union, List
from omegaconf import DictConfig, OmegaConf

from src.data.dataset_factory import get_dataset


class VAEDataModule(pl.LightningDataModule):
    """
    Unified data module for VAE training.
    
    Handles:
    - Dataset instantiation from config
    - Train/val/test splits
    - DataLoader configuration
    - Multi-GPU compatible data loading
    
    Configuration example:
        data:
            target: src.data.datamodule.VAEDataModule
            params:
                dataset:
                    target: src.data.co3d_dataset.CO3DDataset
                    params:
                        root_dir: /path/to/co3d
                        include_plucker: true
                batch_size: 8
                val_split: 0.1
                num_workers: 4
    """
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        batch_size: int = 8,
        val_split: float = 0.1,
        test_split: float = 0.0,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        # Optional: separate configs for train/val/test
        train_dataset_config: Optional[Dict[str, Any]] = None,
        val_dataset_config: Optional[Dict[str, Any]] = None,
        test_dataset_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize VAE DataModule.
        
        Args:
            dataset_config: Configuration for dataset instantiation.
                           Used for all splits unless specific configs provided.
            batch_size: Batch size for all dataloaders
            val_split: Fraction of data to use for validation (0.0 to 1.0)
            test_split: Fraction of data to use for testing (0.0 to 1.0)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU transfer
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            drop_last: Whether to drop last incomplete batch
            persistent_workers: Keep workers alive between epochs
            seed: Random seed for reproducible splits
            train_dataset_config: Optional separate config for training set
            val_dataset_config: Optional separate config for validation set
            test_dataset_config: Optional separate config for test set
        """
        super().__init__()
        
        # Convert OmegaConf to dict if necessary
        if isinstance(dataset_config, DictConfig):
            dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
        
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        
        # Optional separate dataset configs
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.test_dataset_config = test_dataset_config
        
        # Will be set in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        self.save_hyperparameters(ignore=['dataset_config', 'train_dataset_config', 
                                          'val_dataset_config', 'test_dataset_config'])
        
        print(f"[VAEDataModule] Initialized with batch_size={batch_size}, "
              f"val_split={val_split}, num_workers={num_workers}")
    
    def prepare_data(self):
        """
        Download or prepare data. Called only on rank 0.
        
        Override in subclass if dataset requires downloads.
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage.
        
        Args:
            stage: One of 'fit', 'validate', 'test', 'predict', or None
        """
        if stage == "fit" or stage is None:
            self._setup_train_val()
        
        if stage == "test" or stage is None:
            self._setup_test()
    
    def _setup_train_val(self):
        """Set up training and validation datasets."""
        
        # Check if separate configs provided
        if self.train_dataset_config is not None and self.val_dataset_config is not None:
            # Separate datasets for train and val
            print("[VAEDataModule] Using separate train/val dataset configs")
            self.train_dataset = get_dataset(self.train_dataset_config)
            self.val_dataset = get_dataset(self.val_dataset_config)
        else:
            # Single dataset with random split
            print("[VAEDataModule] Creating train/val split from single dataset")
            full_dataset = get_dataset(self.dataset_config)
            
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            test_size = int(total_size * self.test_split)
            train_size = total_size - val_size - test_size
            
            if train_size <= 0:
                raise ValueError(
                    f"Invalid split sizes: total={total_size}, "
                    f"val_split={self.val_split}, test_split={self.test_split} "
                    f"results in train_size={train_size}"
                )
            
            # Use generator for reproducible splits
            generator = torch.Generator().manual_seed(self.seed)
            
            if test_size > 0:
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    full_dataset, 
                    [train_size, val_size, test_size],
                    generator=generator
                )
            else:
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=generator
                )
            
            print(f"[VAEDataModule] Split: train={train_size}, val={val_size}, test={test_size}")
    
    def _setup_test(self):
        """Set up test dataset."""
        if self.test_dataset is not None:
            return  # Already set up from train_val split
        
        if self.test_dataset_config is not None:
            self.test_dataset = get_dataset(self.test_dataset_config)
            print(f"[VAEDataModule] Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test') first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Get prediction dataloader (uses test set)."""
        return self.test_dataloader()
    
    # ==================== Utility Methods ====================
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.
        
        Returns:
            Dictionary with dataset sizes and configuration
        """
        info = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "val_split": self.val_split,
            "test_split": self.test_split,
        }
        
        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)
        
        return info


# Import torch here to avoid circular imports at module level
import torch


class PairedVAEDataModule(VAEDataModule):
    """
    DataModule for paired/multi-view data.
    
    Extends VAEDataModule with support for:
    - Paired image sampling (source + target)
    - View consistency constraints
    - Plucker coordinate loading
    
    Useful for PluckerVAE and multi-view consistent training.
    """
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        batch_size: int = 8,
        val_split: float = 0.1,
        num_workers: int = 4,
        pair_sampling: str = "random",  # "random", "sequential", "fixed"
        include_plucker: bool = True,
        **kwargs
    ):
        """
        Initialize paired data module.
        
        Args:
            dataset_config: Dataset configuration
            batch_size: Batch size
            val_split: Validation split fraction
            num_workers: Number of data loading workers
            pair_sampling: How to sample image pairs:
                          - "random": Random pairs within same object
                          - "sequential": Adjacent frames
                          - "fixed": Fixed pair indices
            include_plucker: Whether to load Plucker coordinates
            **kwargs: Additional arguments passed to VAEDataModule
        """
        # Inject Plucker flag into dataset config
        if "params" not in dataset_config:
            dataset_config["params"] = {}
        dataset_config["params"]["include_plucker"] = include_plucker
        dataset_config["params"]["pair_sampling"] = pair_sampling
        
        super().__init__(
            dataset_config=dataset_config,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            **kwargs
        )
        
        self.pair_sampling = pair_sampling
        self.include_plucker = include_plucker
        
        print(f"[PairedVAEDataModule] pair_sampling={pair_sampling}, include_plucker={include_plucker}")