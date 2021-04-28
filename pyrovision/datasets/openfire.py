# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from pathlib import Path
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional
from PIL import Image
from .utils import download_dataset_from_url

__all__ = ['OpenFire']


url = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/open_fire-dd0b5bb0.zip"
url_sample = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/open_fire_sample-06a07e48.zip"
zip_filename = "open_fire.zip"


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class OpenFire(DatasetFolder):
    """Wildfire image Dataset.

    Args:
        root (string): Root directory of dataset where the ``images``
            and  ``annotations`` folders exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        img_folder (str or Path, optional): Location of image folder. Default: <root>/OpenFire/
        train (bool, optional): If True, returns training subset, else test set.
        sample (bool, optional): If True, use openfire subset with 100 training images and 16 testing.
    """

    def __init__(
            self,
            root: str,
            download=False,
            img_folder=None,
            train=True,
            sample=False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
    ):
        """Init."""
        self.root = root
        self.train = train
        if img_folder is not None:
            self.root = Path(img_folder)

        if download:
            if sample:
                url = url_sample
            download_dataset_from_url(url, self.root)

        if self.train:
            self.path_to_imgs = Path(self.root, 'train')
        else:
            self.path_to_imgs = Path(self.root, 'test')

        super(OpenFire, self).__init__(self.path_to_imgs, loader, None,
                                       transform=transform,
                                       target_transform=target_transform,
                                       is_valid_file=loader)
        self.imgs = self.samples
