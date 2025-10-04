import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Optional
import yaml


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    preds_binary = (predictions > threshold).float()
    targets_binary = (targets > threshold).float()

    intersection = (preds_binary * targets_binary).sum()
    union = preds_binary.sum() + targets_binary.sum()

    dice = (2.0 * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)

    precision = intersection / (preds_binary.sum() + 1e-8)
    recall = intersection / (targets_binary.sum() + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    metrics: Dict[str, float], save_path: str) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                     checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logging.info(f"Checkpoint loaded from {checkpoint_path}")

    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_splits(train_split: float, val_split: float, test_split: float) -> bool:
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Data splits must sum to 1.0, got {total}")
    return True
