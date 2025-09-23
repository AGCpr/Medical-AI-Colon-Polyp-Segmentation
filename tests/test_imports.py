"""Basic import tests to verify package dependencies."""

def test_imports():
    """Test if all required packages can be imported."""
    import torch
    import monai
    import pytorch_lightning
    assert torch.__version__ is not None
    assert monai.__version__ is not None
    assert pytorch_lightning.__version__ is not None

def test_app_imports():
    """Test if application modules can be imported."""
    import app
    import desktop_app
    import model
    import dataset
    assert app is not None
    assert desktop_app is not None
    assert model is not None
    assert dataset is not None