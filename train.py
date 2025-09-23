#!/usr/bin/env python3
"""
Hydra-MONAI-Lightning-FlexibleUNet-ColonPolypSegmentation
Main training script with Hydra configuration management
"""

import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from hydra.utils import instantiate

from model import SegmentationModel
from dataset import SegmentationDataModule


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Set seed for reproducibility
    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)
    
    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Initialize data module
    print("Setting up data module...")
    datamodule = SegmentationDataModule(cfg)
    
    # Initialize model
    print("Setting up model...")
    model = instantiate(
        cfg.model,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    if hasattr(cfg, 'callbacks') and hasattr(cfg.callbacks, 'model_checkpoint'):
        checkpoint_callback = instantiate(cfg.callbacks.model_checkpoint)
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = ModelCheckpoint(
            filename="best_model",
            monitor="val_dice",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    if hasattr(cfg, 'callbacks') and hasattr(cfg.callbacks, 'early_stopping'):
        early_stopping = instantiate(cfg.callbacks.early_stopping)
        callbacks.append(early_stopping)
    
    # Learning rate monitor
    if hasattr(cfg, 'callbacks') and hasattr(cfg.callbacks, 'lr_monitor'):
        lr_monitor = instantiate(cfg.callbacks.lr_monitor)
        callbacks.append(lr_monitor)
    
    # Setup logger
    loggers = []
    
    # WandB logger
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            save_dir=cfg.output_dir
        )
        loggers.append(wandb_logger)
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.experiment_name
    )
    loggers.append(tb_logger)
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.training.num_sanity_val_steps,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=cfg.training.enable_progress_bar,
        enable_model_summary=cfg.training.enable_model_summary,
        fast_dev_run=cfg.training.fast_dev_run,
        overfit_batches=cfg.training.overfit_batches,
        deterministic=cfg.training.deterministic,
        callbacks=callbacks,
        logger=loggers,
        profiler=cfg.training.profiler
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Train model
    print("\nStarting training...")
    trainer.fit(model, datamodule)
    
    # Test model
    print("\nStarting testing...")
    trainer.test(model, datamodule, ckpt_path="best")
    
    # Print best model path
    if hasattr(checkpoint_callback, 'best_model_path'):
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best validation dice: {checkpoint_callback.best_model_score:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()