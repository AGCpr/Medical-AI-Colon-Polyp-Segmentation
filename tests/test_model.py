import pytest
import torch
from omegaconf import OmegaConf
from model import SegmentationModel

# Default model config for testing
MODEL_CONFIG = OmegaConf.create({
    "in_channels": 3,
    "out_channels": 1,
    "backbone": "efficientnet-b0",
    "pre_trained": False,
    "decoder_channels": [256, 128, 64, 32, 16],
    "spatial_dimensions": 2,
})

@pytest.mark.parametrize(
    "optimizer_name, optimizer_class, extra_params",
    [
        ("SGD", torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        ("Adam", torch.optim.Adam, {"lr": 0.001}),
        ("AdamW", torch.optim.AdamW, {"lr": 0.001}),
        ("RMSprop", torch.optim.RMSprop, {"lr": 0.01}),
    ],
)
def test_configure_optimizers(optimizer_name, optimizer_class, extra_params):
    """
    Tests if the SegmentationModel correctly configures various optimizers using Hydra.
    This test verifies that the refactored `configure_optimizers` method can instantiate
    any torch.optim optimizer provided in the configuration, including those that were
    not supported by the previous implementation.
    """
    # Create a mock optimizer config
    optimizer_cfg = {
        "_target_": f"torch.optim.{optimizer_name}",
        **extra_params
    }
    optimizer_conf = OmegaConf.create(optimizer_cfg)

    # Instantiate the model with the test configuration
    model = SegmentationModel(
        model_cfg=MODEL_CONFIG,
        optimizer_cfg=optimizer_conf,
        scheduler_cfg=None  # No scheduler for this test
    )

    # Get the configured optimizer
    optimizer = model.configure_optimizers()

    # Verify that the correct optimizer was created
    assert isinstance(optimizer, optimizer_class), \
        f"Expected optimizer to be {optimizer_class}, but got {type(optimizer)}"

    # Verify that the parameters were passed correctly
    for param_name, param_value in extra_params.items():
        assert optimizer.defaults[param_name] == param_value, \
            f"Expected param '{param_name}' to be {param_value}"

def test_configure_optimizer_with_scheduler():
    """
    Tests if the model correctly configures an optimizer and a learning rate scheduler.
    """
    optimizer_cfg = OmegaConf.create({
        "_target_": "torch.optim.Adam",
        "lr": 0.001
    })
    scheduler_cfg = OmegaConf.create({
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 30,
        "gamma": 0.1
    })

    model = SegmentationModel(
        model_cfg=MODEL_CONFIG,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg
    )

    config = model.configure_optimizers()

    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["optimizer"], torch.optim.Adam)
    assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert config["lr_scheduler"]["monitor"] == "val_dice"