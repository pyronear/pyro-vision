from pathlib import Path

import pytest
from PIL.Image import Image
from torchvision.datasets import VisionDataset

from pyrovision import datasets


@pytest.mark.parametrize(
    "url, max_base_length, expected_name",
    [
        ["https://pyronear.org/img/logo_letters.png", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters.png?height=300", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters.png?height=300&width=400", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters", None, "logo_letters.jpg"],
        ["https://pyronear.org/img/very_long_file_name.png", 10, "very_long_.png"],
    ],
)
def test_get_fname(url, max_base_length, expected_name):
    assert datasets.utils.get_fname(url, max_base_length=max_base_length) == expected_name


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
    test_set = datasets.OpenFire(ds_folder, train=False, download=True, num_samples=num_samples)
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
