import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model import SegmentationModel
    from omegaconf import DictConfig
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required dependencies not installed")
class TestSegmentationModel:

    @pytest.fixture
    def model_config(self):
        return DictConfig({
            'in_channels': 3,
            'out_channels': 1,
            'backbone': 'efficientnet-b4',
            'pre_trained': False,
            'decoder_channels': [256, 128, 64, 32, 16],
            'spatial_dimensions': 2
        })

    @pytest.fixture
    def optimizer_config(self):
        return {
            '_target_': 'torch.optim.Adam',
            'lr': 0.001
        }

    def test_model_initialization(self, model_config, optimizer_config):
        model = SegmentationModel(model_config, optimizer_config)
        assert model is not None
        assert model.model is not None
        assert model.loss_fn is not None

    def test_forward_pass(self, model_config, optimizer_config):
        model = SegmentationModel(model_config, optimizer_config)
        batch_size = 2
        x = torch.randn(batch_size, 3, 320, 320)
        output = model(x)
        assert output.shape == (batch_size, 1, 320, 320)

    def test_training_step(self, model_config, optimizer_config):
        model = SegmentationModel(model_config, optimizer_config)
        batch = {
            'image': torch.randn(2, 3, 320, 320),
            'label': torch.randint(0, 2, (2, 1, 320, 320)).float()
        }
        loss = model.training_step(batch, 0)
        assert loss is not None
        assert loss.requires_grad

    def test_validation_step(self, model_config, optimizer_config):
        model = SegmentationModel(model_config, optimizer_config)
        batch = {
            'image': torch.randn(2, 3, 320, 320),
            'label': torch.randint(0, 2, (2, 1, 320, 320)).float()
        }
        loss = model.validation_step(batch, 0)
        assert loss is not None

    def test_configure_optimizers(self, model_config, optimizer_config):
        model = SegmentationModel(model_config, optimizer_config)
        optimizer = model.configure_optimizers()
        assert optimizer is not None
