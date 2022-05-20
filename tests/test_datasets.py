import json
from pathlib import Path

import pytest
from PIL.Image import Image
from torchvision.datasets import VisionDataset

from pyrovision import datasets


def test_openfire(tmpdir_factory):
    num_samples = 200
    img_folder = str(tmpdir_factory.mktemp("images"))
    root = str(tmpdir_factory.mktemp("root"))

    # Test img_folder argument: wrong type and default (None)
    with pytest.raises(TypeError):
        datasets.OpenFire(root, download=True, img_folder=1)

    ds = datasets.OpenFire(root, download=True, num_samples=num_samples, img_folder=None)
    assert isinstance(ds.img_folder, Path)

    # Working case
    # Test img_folder as Path and str
    train_set = datasets.OpenFire(root=root, train=True, download=True, num_samples=num_samples,
                                  img_folder=Path(img_folder))
    test_set = datasets.OpenFire(root=root, train=False, download=True, num_samples=num_samples,
                                 img_folder=img_folder)
    # Check inherited properties
    assert isinstance(train_set, VisionDataset)

    # Assert valid extensions of every image
    assert (all(sample['name'].rpartition('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']
                for sample in train_set.data))
    assert (all(sample['name'].rpartition('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']
                for sample in test_set.data))

    # Check against number of samples in extract (limit to num_samples)
    datasets.utils.download_url(train_set.url, root, filename='extract.json', verbose=False)
    with open(Path(root).joinpath('extract.json'), 'rb') as f:
        extract = json.load(f)[:num_samples]
    # Test if not more than 15 downloads failed.
    # Change to assertEqual when download issues are resolved
    assert abs((len(train_set) + len(test_set)) - len(extract)) <= 32

    # Check integrity of samples
    img, target = train_set[0]
    assert isinstance(img, Image)
    assert isinstance(target, int)
    assert train_set.class_to_idx[extract[0]['target']] == target

    # Check train/test split
    assert isinstance(train_set, VisionDataset)
    # Check unicity of sample across all splits
    train_paths = [sample['name'] for sample in train_set.data]
    assert all(sample['name'] not in train_paths for sample in test_set.data)
