"""
Visualization utilities for polyp segmentation results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Tuple, Optional, List, Union
from PIL import Image
import os


def plot_segmentation_results(
    images: Union[torch.Tensor, np.ndarray], 
    predictions: Union[torch.Tensor, np.ndarray], 
    targets: Union[torch.Tensor, np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    threshold: float = 0.5,
    show_overlay: bool = True
) -> None:
    """
    Plot segmentation results: input, prediction, target, and overlay
    
    Args:
        images: Input images [B, C, H, W] or [B, H, W, C]
        predictions: Model predictions [B, C, H, W] 
        targets: Ground truth masks [B, C, H, W]
        titles: Optional titles for each sample
        save_path: Path to save the plot
        figsize: Figure size
        threshold: Threshold for binary prediction
        show_overlay: Whether to show overlay visualization
    """
    # Convert tensors to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    n_cols = 4 if show_overlay else 3
    
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(figsize[0], figsize[1] * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Prepare image (handle different channel arrangements)
        if images.shape[1] == 3:  # [B, C, H, W]
            img = np.transpose(images[i], (1, 2, 0))
        else:  # [B, H, W, C] or [B, H, W]
            img = images[i]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
        
        # Normalize image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Get prediction and target masks
        pred_mask = predictions[i, 0] if len(predictions.shape) > 2 else predictions[i]
        target_mask = targets[i, 0] if len(targets.shape) > 2 else targets[i]
        
        # Apply threshold to prediction
        pred_binary = (pred_mask > threshold).astype(np.float32)
        
        # Plot original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Input {i+1}' if titles is None else titles[i])
        axes[i, 0].axis('off')
        
        # Plot prediction
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title(f'Prediction {i+1}')
        axes[i, 1].axis('off')
        
        # Plot target
        axes[i, 2].imshow(target_mask, cmap='gray')
        axes[i, 2].set_title(f'Ground Truth {i+1}')
        axes[i, 2].axis('off')
        
        # Plot overlay if requested
        if show_overlay:
            overlay = create_overlay(img, pred_binary, target_mask)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay {i+1}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_overlay(
    image: np.ndarray, 
    prediction: np.ndarray, 
    target: np.ndarray, 
    pred_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # Red
    target_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # Green
    alpha: float = 0.3
) -> np.ndarray:
    """
    Create overlay visualization of prediction and target on original image
    
    Args:
        image: Original image [H, W, 3]
        prediction: Prediction mask [H, W]
        target: Target mask [H, W]
        pred_color: Color for prediction overlay (RGB)
        target_color: Color for target overlay (RGB)
        alpha: Transparency level
        
    Returns:
        Overlay image [H, W, 3]
    """
    # Ensure image is in [0, 1] range
    if image.max() > 1:
        image = image / 255.0
    
    # Create overlay
    overlay = image.copy()
    
    # Add prediction in red
    pred_mask = prediction > 0.5
    for c in range(3):
        overlay[pred_mask, c] = (1 - alpha) * overlay[pred_mask, c] + alpha * pred_color[c]
    
    # Add target in green (overlap will be yellow)
    target_mask = target > 0.5
    for c in range(3):
        overlay[target_mask, c] = (1 - alpha) * overlay[target_mask, c] + alpha * target_color[c]
    
    return np.clip(overlay, 0, 1)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str = "Dice",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot training history: loss and metrics
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs  
        train_metrics: Training metrics over epochs
        val_metrics: Validation metrics over epochs
        metric_name: Name of the metric being plotted
        save_path: Path to save the plot
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot metrics
    ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}', linewidth=2)
    ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}', linewidth=2)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def plot_model_predictions_batch(
    model,
    dataloader,
    device: str = "cpu",
    num_samples: int = 8,
    save_path: Optional[str] = None,
    threshold: float = 0.5
) -> None:
    """
    Plot model predictions for a batch of samples
    
    Args:
        model: Trained model
        dataloader: DataLoader to get samples from
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
        threshold: Threshold for binary prediction
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch["image"][:num_samples]
    labels = batch["label"][:num_samples]
    
    # Move to device
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.sigmoid(predictions)
    
    # Plot results
    plot_segmentation_results(
        images=images,
        predictions=predictions,
        targets=labels,
        save_path=save_path,
        threshold=threshold,
        figsize=(16, 4 * num_samples // 2)
    )


def calculate_and_plot_metrics_distribution(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> dict:
    """
    Calculate and plot distribution of various metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth masks
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Dictionary with calculated metrics
    """
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    
    # Convert to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    # Calculate metrics per sample
    dice_metric = DiceMetric(include_background=True, reduction="none")
    
    dice_scores = dice_metric(predictions, targets).numpy().flatten()
    
    # Calculate additional metrics
    intersection = (predictions * targets).sum(dim=(2, 3))
    union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou_scores = (intersection / (union + 1e-8)).numpy().flatten()
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Dice distribution
    axes[0, 0].hist(dice_scores, bins=30, alpha=0.7, color='blue')
    axes[0, 0].axvline(dice_scores.mean(), color='red', linestyle='--', 
                      label=f'Mean: {dice_scores.mean():.3f}')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU distribution
    axes[0, 1].hist(iou_scores, bins=30, alpha=0.7, color='green')
    axes[0, 1].axvline(iou_scores.mean(), color='red', linestyle='--',
                      label=f'Mean: {iou_scores.mean():.3f}')
    axes[0, 1].set_title('IoU Score Distribution')
    axes[0, 1].set_xlabel('IoU Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots
    axes[1, 0].boxplot([dice_scores, iou_scores], labels=['Dice', 'IoU'])
    axes[1, 0].set_title('Metrics Box Plot')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1, 1].scatter(dice_scores, iou_scores, alpha=0.6)
    axes[1, 1].set_xlabel('Dice Score')
    axes[1, 1].set_ylabel('IoU Score')
    axes[1, 1].set_title('Dice vs IoU Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics distribution saved to {save_path}")
    
    plt.show()
    
    # Return summary statistics
    return {
        'dice_mean': float(dice_scores.mean()),
        'dice_std': float(dice_scores.std()),
        'dice_min': float(dice_scores.min()),
        'dice_max': float(dice_scores.max()),
        'iou_mean': float(iou_scores.mean()),
        'iou_std': float(iou_scores.std()),
        'iou_min': float(iou_scores.min()),
        'iou_max': float(iou_scores.max())
    }