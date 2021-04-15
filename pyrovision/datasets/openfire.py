# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import zipfile
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional
from PIL import Image

__all__ = ['OpenFire']


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
        train (bool, optional): If True, returns training subset, else test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        threads (int, optional): If download is set to True, use this amount of threads
            for downloading the dataset.
        num_samples (int, optional): Number of samples to download (all by default)
        img_folder (str or Path, optional): Location of image folder. Default: <root>/OpenFire/images
        **kwargs: optional arguments of torchvision.datasets.VisionDataset
    """

    gdrive_file_id = "1rRt7lGLCTVaA6qfdUpGuCBajlOlDQLTF"
    gdrive_file_id_sample = "1u5vA553OrtfiT0IIVvNYjL0iNAhUhB0V"
    filename = "open_fire.zip"

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
        self.root = root
        self.train = train
        if img_folder is not None:
            self.root = Path(img_folder)

        if download:
            self.download(sample)

        if self.train:
            self.path_to_imgs = Path(self.root, 'train')
        else:
            self.path_to_imgs = Path(self.root, 'test')

        super(OpenFire, self).__init__(self.path_to_imgs, loader, None,
                                       transform=transform,
                                       target_transform=target_transform,
                                       is_valid_file=loader)
        self.imgs = self.samples

    def download(self, sample):
        """
        Download dataset"""
        print('Downloading OpenFire ...')
        if sample:
            gdrive_file_id = self.gdrive_file_id_sample
        else:
            gdrive_file_id = self.gdrive_file_id

        download_file_from_google_drive(gdrive_file_id, '.', self.filename)
        print("Unziping ...")
        with zipfile.ZipFile(self.filename, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print('Done!')
