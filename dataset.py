"""
PyTorch Lightning DataModule for Kvasir-SEG Polyp Segmentation Dataset
"""

import os
import pytorch_lightning as pl
from typing import Optional, List, Dict, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Compose
from hydra.utils import instantiate
from omegaconf import DictConfig

from custom_dataset import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling Kvasir-SEG dataset
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.transform_cfg = cfg.transforms
        
        # Data paths
        self.image_dir = cfg.data.image_dir
        self.mask_dir = cfg.data.mask_dir
        self.png_image_dir = cfg.data.png_image_dir
        self.png_mask_dir = cfg.data.png_mask_dir
        
        # Data splits
        self.train_split = cfg.data.train_split
        self.val_split = cfg.data.val_split
        self.test_split = cfg.data.test_split
        
        # Data loading parameters
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.pin_memory = cfg.data.pin_memory
        self.persistent_workers = cfg.data.persistent_workers
        
        # File lists
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Transforms
        self.train_transforms = None
        self.val_transforms = None
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self) -> None:
        """
        Download or prepare data if needed.
        This method is called only from a single process.
        """
        # Check if data directories exist
        if not os.path.exists(self.image_dir) and not os.path.exists(self.png_image_dir):
            raise FileNotFoundError(
                f"Neither {self.image_dir} nor {self.png_image_dir} exists. "
                "Please download the Kvasir-SEG dataset."
            )
        
        print(f"Data directories found.")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for train, validation, and test.
        This method is called from every process.
        """
        # Choose data directory (prefer PNG if available)
        if os.path.exists(self.png_image_dir) and os.path.exists(self.png_mask_dir):
            image_dir = self.png_image_dir
            mask_dir = self.png_mask_dir
            print("Using PNG format dataset")
        else:
            image_dir = self.image_dir
            mask_dir = self.mask_dir
            print("Using original format dataset")
        
        # Get file list
        image_files = self._get_file_list(image_dir, self.data_cfg.image_extensions)
        mask_files = self._get_file_list(mask_dir, self.data_cfg.mask_extensions)
        
        # Match image and mask files
        file_pairs = self._match_files(image_files, mask_files, image_dir, mask_dir)
        
        print(f"Found {len(file_pairs)} image-mask pairs")
        
        # Split data
        if len(file_pairs) == 0:
            raise ValueError("No matching image-mask pairs found!")
        
        # First split: train + val, test
        train_val_files, test_files = train_test_split(
            file_pairs,
            test_size=self.test_split,
            random_state=self.cfg.seed if hasattr(self.cfg, 'seed') else 42,
            shuffle=True
        )
        
        # Second split: train, val
        val_size = self.val_split / (self.train_split + self.val_split)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_size,
            random_state=self.cfg.seed if hasattr(self.cfg, 'seed') else 42,
            shuffle=True
        )
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Setup transforms
        self._setup_transforms()
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SegmentationDataset(
                file_pairs=train_files,
                transforms=self.train_transforms
            )
            self.val_dataset = SegmentationDataset(
                file_pairs=val_files,
                transforms=self.val_transforms
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = SegmentationDataset(
                file_pairs=test_files,
                transforms=self.val_transforms
            )
    
    def _get_file_list(self, directory: str, extensions: List[str]) -> List[str]:
        """Get list of files with specified extensions"""
        files = []
        if not os.path.exists(directory):
            return files
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                files.append(file)
        
        return sorted(files)
    
    def _match_files(self, image_files: List[str], mask_files: List[str], 
                    image_dir: str, mask_dir: str) -> List[Dict[str, str]]:
        """Match image files with corresponding mask files"""
        file_pairs = []
        
        for img_file in image_files:
            # Get base name without extension
            base_name = os.path.splitext(img_file)[0]
            
            # Look for corresponding mask
            mask_file = None
            for mask in mask_files:
                if os.path.splitext(mask)[0] == base_name:
                    mask_file = mask
                    break
            
            if mask_file:
                file_pairs.append({
                    "image": os.path.join(image_dir, img_file),
                    "label": os.path.join(mask_dir, mask_file)
                })
        
        return file_pairs
    
    def _setup_transforms(self) -> None:
        """Setup MONAI transforms for training and validation"""
        if hasattr(self.transform_cfg, 'train_transforms'):
            # Use configured transforms
            train_transform_list = [instantiate(t) for t in self.transform_cfg.train_transforms]
            self.train_transforms = Compose(train_transform_list)
        else:
            # Use default transforms
            self.train_transforms = self._get_default_train_transforms()
        
        if hasattr(self.transform_cfg, 'val_transforms'):
            # Use configured transforms
            val_transform_list = [instantiate(t) for t in self.transform_cfg.val_transforms]
            self.val_transforms = Compose(val_transform_list)
        else:
            # Use default transforms
            self.val_transforms = self._get_default_val_transforms()
    
    def _get_default_train_transforms(self) -> Compose:
        """Default training transforms"""
        from monai.transforms import (
            LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityRanged,
            RandRotated, RandFlipd, EnsureTyped
        )
        
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=self.data_cfg.image_size),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            RandRotated(keys=["image", "label"], range_x=0.2, prob=0.3),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            EnsureTyped(keys=["image", "label"])
        ])
    
    def _get_default_val_transforms(self) -> Compose:
        """Default validation transforms"""
        from monai.transforms import (
            LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityRanged, EnsureTyped
        )
        
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=self.data_cfg.image_size),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            EnsureTyped(keys=["image", "label"])
        ])
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )