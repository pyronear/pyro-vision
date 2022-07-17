# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder

from .utils import download_url, download_urls, parallel

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ["OpenFire"]


DEFAULT_EXT = "jpg"
IMG_EXTS = ["jpg", "jpeg", "png", "gif"]


def _resolve_img_extension(url: str) -> str:
    url_lower = url.lower()
    for ext in IMG_EXTS:
        if f".{ext}" in url_lower:
            return ext
    return DEFAULT_EXT


def _validate_img_file(file_path: Union[str, Path]) -> bool:
    try:
        Image.open(file_path, mode="r").convert("RGB")
    except Exception:
        return False
    return True


class OpenFire(ImageFolder):
    """Implements an image classification dataset for wildfire detection, collected from web searches.

    >>> from pyrovision.datasets import OpenFire
    >>> train_set = OpenFire("path/to/your/folder", train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        root: Root directory where 'OpenFire' is located.
        train: If True, returns training subset, else validation set.
        download: If True, downloads the dataset from the internet and puts it in root directory. If dataset is
            already downloaded, it is not downloaded again.
        num_samples: Number of samples to download (all by default)
        num_threads: If download is set to True, use this amount of threads for downloading the dataset.
        prefetch_fn: optional function that will be applied to all images before data loading
        **kwargs: optional arguments of torchvision.datasets.VisionDataset
    """

    TRAIN = (
        "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/openfire_train-d912c0b4.json",
        "d912c0b4c4fb89f482c1ad8e4b47c79202efbeedc832a29f779944afd17118be",
    )
    VAL = (
        "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/openfire_val-31235919.json",
        "31235919c7ed278731f6511eae42c7d27756a88e86a9b32d7b1ff105dc31097d",
    )
    CLASSES = ["Wildfire"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        validate_images: bool = True,
        num_samples: Optional[int] = None,
        num_threads: Optional[int] = None,
        prefetch_fn: Optional[Callable[[str, str], None]] = None,
        **kwargs: Any,
    ) -> None:
        # Folder management
        _root = Path(root, self.__class__.__name__)
        _root.mkdir(parents=True, exist_ok=True)
        _root = _root.joinpath("train" if train else "val")
        self.train = train

        # Images
        img_folder = _root.joinpath("images")
        url, sha256 = self.TRAIN if train else self.VAL
        extract_name = url.rpartition("/")[-1]
        extract_path = _root.joinpath(extract_name)

        # Download & verify the subset URLs
        if download:
            # Check whether the file exist
            if not extract_path.is_file():
                download_url(url, _root, filename=extract_name, verbose=False)
            # Check integrity
            with open(extract_path, "rb") as f:
                sha_hash = hashlib.sha256(f.read()).hexdigest()

            assert sha_hash == sha256, f"corrupted download: {extract_path}"

        if not extract_path.is_file():
            raise FileNotFoundError("Extract not found. You can use download=True to download it.")
        with open(extract_path, "rb") as f:
            extract = json.load(f)

        # Only consider num_samples
        if isinstance(num_samples, int):
            num_cat = len(extract)
            cat_size = num_samples // num_cat
            final_size = num_samples - (num_cat - 1) * cat_size
            cats = list(extract.keys())
            for k in cats[:-1]:
                extract[k] = extract[k][:cat_size]
            extract[cats[-1]] = extract[cats[-1]][:final_size]

        file_names = {
            label: [f"{idx:04d}.{_resolve_img_extension(url)}" for idx, url in enumerate(v)]
            for label, v in extract.items()
        }

        if download:
            # Download the images
            for label, urls in extract.items():
                _folder = img_folder.joinpath(label)
                _folder.mkdir(parents=True, exist_ok=True)
                # Prepare URL and filenames for multi-processing
                entries = [
                    (url, file_name)
                    for url, file_name in zip(urls, file_names[label])
                    if not _folder.joinpath(file_name).is_file()
                ]
                # Use multiple threads to speed up download
                if len(entries) > 0:
                    download_urls(entries, _folder, num_threads=num_threads)

        num_files = sum(len(v) for _, v in extract.items())

        # Load & verify the images
        existing_paths = [
            img_folder.joinpath(label, file_name)
            for label, file_names in file_names.items()
            for file_name in file_names
            if img_folder.joinpath(label, file_name).is_file()
        ]

        if len(existing_paths) == 0:
            raise FileNotFoundError("Images not found. You can use download=True to download them.")
        elif len(existing_paths) < num_files:
            warnings.warn(f"number of files that couldn't be found: {num_files - len(existing_paths)}")

        # Enforce image validation
        num_files = len(existing_paths)

        # Check that image can be read
        is_valid = parallel(_validate_img_file, existing_paths, desc="Verifying images", progress=True, leave=False)
        num_valid = sum(is_valid)
        if num_valid < num_files:
            warnings.warn(f"number of unreadable files: {num_files - num_valid}")
        # Remove invalid files (so that they can be restored upon next download)
        parallel(os.remove, [_path for _path, _valid in zip(existing_paths, is_valid) if not _valid])

        # Allow prefetch operations
        if prefetch_fn is not None:
            # Create prefetched folders
            prefetch_folder = Path(root, self.__class__.__name__, "prefetch", "train" if train else "val", "images")
            for label in extract:
                prefetch_folder.joinpath(label).mkdir(parents=True, exist_ok=True)
            # Perform the prefetching operation
            parallel(
                prefetch_fn,
                [
                    (_path, prefetch_folder.joinpath(_path.parent.name, _path.name))
                    for _path, _valid in zip(existing_paths, is_valid)
                    if _valid
                ],
                desc="Prefetching images",
                progress=True,
                leave=False,
            )
            img_folder = prefetch_folder

        super().__init__(img_folder, **kwargs)

    def extra_repr(self) -> str:
        return f"train={self.train}"
