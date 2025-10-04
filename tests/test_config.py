import pytest
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestConfiguration:

    def test_main_config_exists(self):
        assert os.path.exists('config/config.yaml')

    def test_main_config_valid_yaml(self):
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert 'project_name' in config
        assert 'experiment_name' in config

    def test_data_config_exists(self):
        assert os.path.exists('config/data.yaml')

    def test_data_config_valid(self):
        with open('config/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert 'batch_size' in config
        assert 'train_split' in config
        assert 'val_split' in config
        assert 'test_split' in config

        total_split = config['train_split'] + config['val_split'] + config['test_split']
        assert abs(total_split - 1.0) < 0.01, "Data splits should sum to 1.0"

    def test_model_config_exists(self):
        assert os.path.exists('config/model/unet.yaml')

    def test_model_config_valid(self):
        with open('config/model/unet.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert '_target_' in config
        assert 'in_channels' in config
        assert 'out_channels' in config

    def test_training_config_exists(self):
        assert os.path.exists('config/training/training.yaml')

    def test_training_config_valid(self):
        with open('config/training/training.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert 'optimizer' in config
        assert 'max_epochs' in config

    def test_callbacks_config_exists(self):
        assert os.path.exists('config/callbacks/callbacks.yaml')

    def test_transforms_config_exists(self):
        assert os.path.exists('config/transforms/transforms.yaml')

    def test_transforms_config_valid(self):
        with open('config/transforms/transforms.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert 'train_transforms' in config
        assert 'val_transforms' in config
        assert isinstance(config['train_transforms'], list)
        assert isinstance(config['val_transforms'], list)
