# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
from typing import Any, Callable, Optional, Tuple
from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import glob


__all__ = ['OpenFire']


class OpenFire(VisionDataset):
    """Wildfire image Dataset.

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, returns training subset, else test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        sample (bool, optional): If True, use openfire subset with 64 training images and 16 testing.
    """

    urls = {'ds': 'https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/open_fire.zip',
            'sample': 'https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/open_fire_sample.zip'
            }

    md5s = {'ds': '5a532853ac17dc43ed7dd4a97a15d715',
            'sample': '31117230cceb029d557a1981f0f30cf7'
            }

    classes = ['0 - No Fire', '1 - Fire']

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            sample: bool = False
    ) -> None:
        """Init."""

        super(OpenFire, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        self.train = train  # training set or test set
        self.sample = sample  # sample dataset for test purpose

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data = self._load_data()

    def _load_data(self):
        """Get images."""
        image_file = 'train' if self.train else 'test'
        data = os.path.join(self.raw_folder, image_file)

        return glob.glob(data + '/**/*g', recursive=True)  # get all jpg, jpeg, png

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get Item
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]  # get image
        target = os.path.normpath(img)  # get target from image path
        target = int(target.split(os.sep)[-2])

        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Length"""
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root)

    def _check_exists(self) -> bool:

        return check_integrity(os.path.join(self.raw_folder, self.filename))

    def download(self) -> None:
        """Download the OpenFire data if it doesn't exist already."""
        # download files
        if self.sample:
            self.url = self.urls['sample']
            md5 = self.md5s['sample']
        else:
            self.url = self.urls['ds']
            md5 = self.md5s['ds']

        path = os.path.normpath(self.url)
        self.filename = path.split(os.sep)[-1]

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        try:
            download_and_extract_archive(
                self.url, download_root=self.raw_folder,
                filename=self.filename,
                md5=md5
            )
        except URLError as error:
            print(
                "Failed to download (trying next):\n{}".format(error)
            )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
