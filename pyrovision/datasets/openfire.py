# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from PIL import Image, ImageFile
from torchvision.datasets import VisionDataset

from .utils import download_url, download_urls

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


def _find_ext_files(dir, ext):
    # borrowed from
    # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = _find_ext_files(dir, ext)
        subfolders.extend(sf)
        files.extend(f)  # type: ignore[arg-type]
    return subfolders, files


def _validate_img_file(file_path: Union[str, Path]) -> bool:
    try:
        Image.open(file_path, mode="r").convert("RGB")
    except Exception:
        pass
    return True


class OpenFire(VisionDataset):
    """Implements an image classification dataset for wildfire detection, collected from web searches.

    >>> from pyrovision.datasets import OpenFire
    >>> train_set = OpenFire("path/to/your/folder", train=True, download=True)
    >>> img, target = train_set[0]

    Args:
        root: Root directory where 'OpenFire' is located.
        train: If True, returns training subset, else validation set.
        download: If true, downloads the dataset from the internet and puts it in root directory. If dataset is
            already downloaded, it is not downloaded again.
        num_samples: Number of samples to download (all by default)
        num_threads: If download is set to True, use this amount of threads for downloading the dataset.
        **kwargs: optional arguments of torchvision.datasets.VisionDataset
    """

    TRAIN = (
        "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/openfire_train-92e9cf25.json",
        "92e9cf25a8600e5e511cee5f2d6d90dba84a701bcd3d7118641b3651d335493c",
    )
    VAL = (
        "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/openfire_val-0a514867.json",
        "0a514867e052a5149996afee3acf13b7976c03c26b589c5908efe700e496105f",
    )
    CLASSES = [False, True]

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        num_samples: Optional[int] = None,
        num_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Folder management
        _root = Path(root, self.__class__.__name__)
        _root.mkdir(parents=True, exist_ok=True)
        _root = _root.joinpath("train" if train else "val")
        super().__init__(_root, **kwargs)
        self.train = train

        # Images
        self.img_folder = _root.joinpath("images")
        subset_url = self.TRAIN if train else self.VAL
        extract_name = subset_url[0].rpartition("/")[-1]
        extract_path = _root.joinpath(extract_name)

        # Download & verify the subset URLs
        if download:
            # Check whether the file exist
            if not extract_path.is_file():
                download_url(subset_url[0], self.root, filename=extract_name, verbose=False)
            # Check integrity
            with open(extract_path, "rb") as f:
                sha_hash = hashlib.sha256(f.read()).hexdigest()

            assert sha_hash == subset_url[1], f"corrupted download: {extract_path}"

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

        if download:
            # Download the images
            for label, urls in extract.items():
                _folder = self.img_folder.joinpath(label)
                _folder.mkdir(parents=True, exist_ok=True)
                # Prepare URL and filenames for multi-processing
                file_names = [f"{idx:04d}.{_resolve_img_extension(url)}" for idx, url in enumerate(urls)]
                entries = [
                    (url, file_name)
                    for url, file_name in zip(urls, file_names)
                    if not _folder.joinpath(file_name).is_file()
                ]
                # Use multiple threads to speed up download
                if len(entries) > 0:
                    download_urls(entries, _folder, num_threads=num_threads)

        _, files = _find_ext_files(self.img_folder, [f".{ext}" for ext in IMG_EXTS])
        if len(files) == 0:
            raise FileNotFoundError("Images not found. You can use download=True to download them.")

        num_files = sum(len(v) for _, v in extract.items())

        # Load & verify the images
        self.data = [
            (os.path.join(label, file), int(label))
            for label in extract
            for file in os.listdir(self.img_folder.joinpath(label))
            if self.img_folder.joinpath(label, file).is_file()
        ]

        if len(self.data) < num_files:
            warnings.warn(f"number of files that couldn't be found: {num_files - len(self.data)}")

        num_files = len(self.data)

        # Check that image can be read
        self.data = [
            (file_path, label)
            for file_path, label in self.data
            if _validate_img_file(self.img_folder.joinpath(file_path))
        ]

        if len(self.data) < num_files:
            warnings.warn(f"number of unreadable files: {num_files - len(self.data)}")

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Getter function"""

        # Load image
        img = Image.open(self.img_folder.joinpath(self.data[idx][0]), mode="r").convert("RGB")
        # Load bboxes & encode label
        target = self.data[idx][1]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return f"train={self.train}"
