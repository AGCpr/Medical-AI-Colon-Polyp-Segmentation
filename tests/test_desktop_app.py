import os
import pytest
import torch
import tkinter as tk
from unittest.mock import MagicMock, patch
from PIL import Image
from desktop_app import DesktopApp, SegmentationModel

# Mock the SegmentationModel class for testing purposes
class MockModel(torch.nn.Module):
    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams is not None else {}
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return x

@pytest.fixture
def root():
    """Fixture to create and destroy a Tkinter root window for tests."""
    r = tk.Tk()
    yield r
    r.destroy()

@pytest.fixture
def app(root):
    """Fixture to create an instance of DesktopApp."""
    # Prevent messagebox from popping up during tests
    with patch('desktop_app.messagebox.showwarning') as mock_showwarning:
        app_instance = DesktopApp(root)
        app_instance.status_var.set = MagicMock() # Mock status bar updates
        yield app_instance

def test_preprocess_uses_model_specific_input_size(app):
    """
    Verify that the _preprocess method resizes images to the dimensions
    specified in the loaded model's hyperparameters, not a hardcoded default.
    """
    # 1. Define custom hyperparameters for the mock model
    custom_input_size = (256, 256)
    mock_hparams = {"spatial_size": list(custom_input_size)}

    # 2. Create a mock model with these hyperparameters
    mock_model = MockModel(hparams=mock_hparams)

    # 3. Patch the model loading and checkpoint path
    with patch('desktop_app.SegmentationModel.load_from_checkpoint', return_value=mock_model) as mock_load, \
         patch('os.path.exists', return_value=True):

        # 4. Simulate loading the model
        app.ckpt_path.set("fake/path/to/model.ckpt")
        app._load_model()

        # 5. Verify that the model's input size was correctly extracted
        assert app.model_input_size == custom_input_size, \
            f"Expected model_input_size to be {custom_input_size}, but got {app.model_input_size}"

        # 6. Create a dummy image for preprocessing
        dummy_image = Image.new('RGB', (600, 400))

        # 7. Preprocess the image
        processed_tensor = app._preprocess(dummy_image)

        # 8. Assert the output tensor has the correct shape (C, H, W)
        # The tensor shape is (1, 3, H, W) -> (N, C, H, W)
        _, _, height, width = processed_tensor.shape
        assert (height, width) == custom_input_size, \
            f"Expected processed image size to be {custom_input_size}, but got {(height, width)}"

def test_preprocess_fallback_to_default_size(app):
    """
    Verify that if a model has no `spatial_size` hparam, the app falls
    back to the default size.
    """
    # 1. Create a mock model with no relevant hparams
    mock_model = MockModel(hparams={})

    # 2. Patch model loading and mock message box
    with patch('desktop_app.SegmentationModel.load_from_checkpoint', return_value=mock_model) as mock_load, \
         patch('desktop_app.messagebox.showwarning') as mock_showwarning, \
         patch('os.path.exists', return_value=True):

        # 3. Simulate loading the model
        app.ckpt_path.set("fake/path/to/model.ckpt")
        app._load_model()

        # 4. Assert that the fallback default size is used and a warning was shown
        default_size = (320, 320)
        assert app.model_input_size == default_size
        mock_showwarning.assert_called_once()

        # 5. Preprocess a dummy image
        dummy_image = Image.new('RGB', (600, 400))
        processed_tensor = app._preprocess(dummy_image)

        # 6. Assert the output tensor has the default shape
        _, _, height, width = processed_tensor.shape
        assert (height, width) == default_size, \
            f"Expected processed image size to be {default_size}, but got {(height, width)}"