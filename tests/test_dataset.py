import pytest
import sys
import os
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from custom_dataset import SegmentationDataset
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required dependencies not installed")
class TestSegmentationDataset:

    @pytest.fixture
    def temp_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'images')
            mask_dir = os.path.join(tmpdir, 'masks')
            os.makedirs(img_dir)
            os.makedirs(mask_dir)

            file_pairs = []
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
                mask = Image.fromarray(np.random.randint(0, 255, (320, 320), dtype=np.uint8))

                img_path = os.path.join(img_dir, f'image_{i}.png')
                mask_path = os.path.join(mask_dir, f'image_{i}.png')

                img.save(img_path)
                mask.save(mask_path)

                file_pairs.append({
                    'image': img_path,
                    'label': mask_path
                })

            yield file_pairs

    def test_dataset_initialization(self, temp_dataset):
        dataset = SegmentationDataset(temp_dataset, transforms=None)
        assert len(dataset) == 5

    def test_dataset_getitem(self, temp_dataset):
        dataset = SegmentationDataset(temp_dataset, transforms=None)
        item = dataset[0]
        assert 'image' in item
        assert 'label' in item

    def test_dataset_file_validation(self):
        invalid_pairs = [
            {'image': '/nonexistent/image.png', 'label': '/nonexistent/mask.png'}
        ]
        dataset = SegmentationDataset(invalid_pairs, transforms=None)
        assert len(dataset) == 0

    def test_get_file_paths(self, temp_dataset):
        dataset = SegmentationDataset(temp_dataset, transforms=None)
        paths = dataset.get_file_paths(0)
        assert 'image' in paths
        assert 'label' in paths

    def test_get_stats(self, temp_dataset):
        dataset = SegmentationDataset(temp_dataset, transforms=None)
        stats = dataset.get_stats()
        assert stats['total_samples'] == 5
        assert stats['has_transforms'] == False
