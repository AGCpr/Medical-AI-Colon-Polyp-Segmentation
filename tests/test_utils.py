import pytest
import torch
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils import (
        compute_metrics, validate_splits, count_parameters,
        get_device, load_config
    )
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


@pytest.mark.skipif(not HAS_UTILS, reason="Utils module not available")
class TestUtils:

    def test_compute_metrics(self):
        predictions = torch.tensor([[0.8, 0.2], [0.6, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        metrics = compute_metrics(predictions, targets, threshold=0.5)

        assert 'dice' in metrics
        assert 'iou' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        assert 0.0 <= metrics['dice'] <= 1.0
        assert 0.0 <= metrics['iou'] <= 1.0

    def test_validate_splits_valid(self):
        assert validate_splits(0.7, 0.15, 0.15) == True

    def test_validate_splits_invalid(self):
        with pytest.raises(ValueError):
            validate_splits(0.7, 0.2, 0.2)

    def test_count_parameters(self):
        model = torch.nn.Linear(10, 5)
        count = count_parameters(model)
        assert count == 55

    def test_get_device(self):
        device = get_device()
        assert device.type in ['cuda', 'cpu', 'mps']

    def test_load_config(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test_key': 'test_value'}, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config['test_key'] == 'test_value'
        finally:
            os.unlink(config_path)

    def test_load_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')
