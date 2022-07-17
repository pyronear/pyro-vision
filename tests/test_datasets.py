from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode, resize

from pyrovision import datasets


def test_openfire(tmpdir_factory):
    num_samples = 100
    ds_folder = str(tmpdir_factory.mktemp("datasets"))

    with pytest.raises(FileNotFoundError):
        datasets.OpenFire(ds_folder, download=False)

    train_set = datasets.OpenFire(ds_folder, download=True, num_samples=num_samples)
    assert isinstance(train_set.root, Path)

    test_set = datasets.OpenFire(ds_folder, train=False, download=True, num_samples=num_samples)
    # Check inherited properties
    assert isinstance(train_set, ImageFolder)

    # Assert valid extensions of every image
    assert all(sample[0].rpartition(".")[-1] in ["jpg", "jpeg", "png", "gif"] for sample in train_set.samples)
    assert all(sample[0].rpartition(".")[-1] in ["jpg", "jpeg", "png", "gif"] for sample in test_set.samples)

    # Check against number of samples in extract (limit to num_samples)
    assert abs(len(train_set) - num_samples) <= 5
    assert abs(len(test_set) - num_samples) <= 5

    # Check integrity of samples
    img, target = train_set[0]
    assert isinstance(img, Image.Image)
    assert isinstance(target, int) and 0 <= target <= len(train_set.CLASSES)

    # Test prefetching
    prefetch_size = 512

    def prefetch_fn(img_paths):
        # Unpack paths
        src_path, dest_path = img_paths
        img = Image.open(src_path, mode="r").convert("RGB")
        # Resize & save
        if all(dim > prefetch_size for dim in img.size):
            resized_img = resize(img, prefetch_size, interpolation=InterpolationMode.BILINEAR)
            resized_img.save(dest_path)
        # Copy
        else:
            copyfile(src_path, dest_path)

    num_samples = len(train_set)
    train_set = datasets.OpenFire(
        root=ds_folder,
        train=True,
        download=False,
        num_samples=num_samples,
        prefetch_fn=prefetch_fn,
    )

    # It's still a mystery why those two values aren't equal
    assert abs(num_samples - len(train_set)) <= 5
    assert "prefetch/" in train_set.samples[0][0]
