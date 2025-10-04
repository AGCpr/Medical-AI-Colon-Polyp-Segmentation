"""
Custom MONAI-compatible dataset class for polyp segmentation
"""

import os
import logging
from typing import Dict, List, Optional, Callable, Any
from torch.utils.data import Dataset
from monai.data import Dataset as MonaiDataset

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """
    Custom dataset class for polyp segmentation that works with MONAI transforms
    """
    
    def __init__(
        self,
        file_pairs: List[Dict[str, str]],
        transforms: Optional[Callable] = None
    ):
        """
        Initialize dataset
        
        Args:
            file_pairs: List of dictionaries with 'image' and 'label' keys containing file paths
            transforms: MONAI transforms to apply
        """
        self.file_pairs = file_pairs
        self.transforms = transforms
        
        # Validate file pairs
        self._validate_files()
        
    def _validate_files(self) -> None:
        """Validate that all files exist"""
        valid_pairs = []
        
        for pair in self.file_pairs:
            if not isinstance(pair, dict):
                continue
                
            image_path = pair.get('image')
            label_path = pair.get('label')
            
            if not image_path or not label_path:
                continue
                
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                continue

            if not os.path.exists(label_path):
                logger.warning(f"Label file not found: {label_path}")
                continue
                
            valid_pairs.append(pair)
        
        self.file_pairs = valid_pairs
        logger.info(f"Dataset initialized with {len(self.file_pairs)} valid file pairs")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with 'image' and 'label' tensors
        """
        if idx >= len(self.file_pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.file_pairs)}")
        
        # Get file paths
        file_pair = self.file_pairs[idx]
        
        # Create data dictionary for MONAI transforms
        data = {
            'image': file_pair['image'],
            'label': file_pair['label']
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            try:
                data = self.transforms(data)
            except Exception as e:
                logger.error(f"Error applying transforms to {file_pair['image']}: {e}")
                raise e
        
        return data
    
    def get_file_paths(self, idx: int) -> Dict[str, str]:
        """
        Get file paths for a given index
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with image and label file paths
        """
        if idx >= len(self.file_pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.file_pairs)}")
        
        return self.file_pairs[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_samples': len(self.file_pairs),
            'has_transforms': self.transforms is not None
        }


class MonaiSegmentationDataset(MonaiDataset):
    """
    MONAI native dataset wrapper for better integration
    """
    
    def __init__(
        self,
        file_pairs: List[Dict[str, str]],
        transform: Optional[Callable] = None,
        cache_rate: float = 0.0,
        num_workers: int = 0
    ):
        """
        Initialize MONAI dataset
        
        Args:
            file_pairs: List of dictionaries with 'image' and 'label' keys
            transform: MONAI transforms to apply
            cache_rate: Cache rate for caching transformed data
            num_workers: Number of workers for caching
        """
        # Validate files first
        valid_pairs = []
        for pair in file_pairs:
            if (isinstance(pair, dict) and 
                'image' in pair and 'label' in pair and
                os.path.exists(pair['image']) and 
                os.path.exists(pair['label'])):
                valid_pairs.append(pair)

        logger.info(f"MONAI dataset initialized with {len(valid_pairs)} valid file pairs")
        
        super().__init__(
            data=valid_pairs,
            transform=transform
        )
        
        # Setup caching if requested
        if cache_rate > 0:
            self.set_cache_enabled(
                cache_rate=cache_rate,
                num_workers=num_workers
            )