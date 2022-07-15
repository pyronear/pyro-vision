from pathlib import Path

import pytest
from PIL.Image import Image
from torchvision.datasets import VisionDataset

from pyrovision import datasets


def test_openfire(tmpdir_factory):
    num_samples = 100
    ds_folder = str(tmpdir_factory.mktemp("datasets"))

    with pytest.raises(FileNotFoundError):
        datasets.OpenFire(ds_folder, download=False)

    ds = datasets.OpenFire(ds_folder, download=True, num_samples=num_samples)
    assert isinstance(ds.img_folder, Path)

    # Working case
    # Test img_folder as Path and str
    train_set = datasets.OpenFire(
        root=ds_folder,
        train=True,
        download=True,
        num_samples=num_samples,
    )
    test_set = datasets.OpenFire(root=ds_folder, train=False, download=True, num_samples=num_samples)
    # Check inherited properties
    assert isinstance(train_set, VisionDataset)

    # Assert valid extensions of every image
    assert all(sample[0].rpartition(".")[-1] in ["jpg", "jpeg", "png", "gif"] for sample in train_set.data)
    assert all(sample[0].rpartition(".")[-1] in ["jpg", "jpeg", "png", "gif"] for sample in test_set.data)

    # Check against number of samples in extract (limit to num_samples)
    assert abs(len(train_set) - num_samples) <= 5
    assert abs(len(test_set) - num_samples) <= 5

    # Check integrity of samples
    img, target = train_set[0]
    assert isinstance(img, Image)
    assert isinstance(target, int) and 0 <= target <= len(train_set.CLASSES)
