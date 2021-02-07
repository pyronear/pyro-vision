# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from pathlib import Path
import warnings
import json
from PIL import Image, ImageFile

from torchvision.datasets import VisionDataset
from .utils import download_url, download_urls

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['OpenFire']


class OpenFire(VisionDataset):
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

    url = 'https://gist.githubusercontent.com/frgfm/f53b4f53a1b2dc3bb4f18c006a32ec0d/raw/c0351134e333710c6ce0c631af5198e109ed7a92/openfire_binary.json'  # noqa: E501
    classes = [False, True]

    def __init__(self, root, train=True, download=False, threads=None, num_samples=None,
                 img_folder=None, **kwargs):
        super(OpenFire, self).__init__(root, **kwargs)
        self.train = train
        if img_folder is None:
            self.img_folder = Path(self.root, self.__class__.__name__, 'images')
        else:
            self.img_folder = Path(img_folder)

        if download:
            self.download(threads, num_samples)

        # Load appropriate subset
        extract = [sample for sample in self.get_extract(num_samples)
                   if sample['is_test'] == (not train)]

        # Verify samples
        self.data = self._verify_samples(extract)

    @property
    def _images(self):
        return self.img_folder

    @property
    def _annotations(self):
        return Path(self.root, self.__class__.__name__, 'annotations')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, idx):
        """ Getter function

        Args:
            index (int): Index
        Returns:
            img (torch.Tensor<float>): image tensor
            target (int): dictionary of bboxes and labels' tensors
        """

        # Load image
        img = Image.open(self._images.joinpath(self.data[idx]['name']), mode='r').convert('RGB')
        # Load bboxes & encode label
        target = self.class_to_idx[self.data[idx]['target']]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self, threads=None, num_samples=None):
        """ Download images from a specific extract

        Args:
            threads (int, optional): number of threads used for parallel downloading
            num_samples (int, optional): if specified, takes first num_samples from extract
        """

        # Download extract of samples
        self._download_extract()

        # Load only the number of specified samples
        extract = self.get_extract(num_samples)

        # Download the corresponding images
        self._download_images(extract, threads)

        # Verify download
        _ = self._verify_samples(extract)

        print('Done!')

    def _download_extract(self):
        """ Download extract file from URL """

        self._annotations.mkdir(parents=True, exist_ok=True)

        # Download annotations
        download_url(self.url, self._annotations, filename=self.url.rpartition('/')[-1], verbose=False)

    def get_extract(self, num_samples=None):
        """ Load extract into memory

        Args:
            num_samples (int, optional): if specified, takes first num_samples from extract
        Returns:
            extract (list<dict>): loaded extract
        """

        # Check extract existence
        file_path = self._annotations.joinpath(self.url.rpartition('/')[-1])
        if not file_path.is_file():
            raise RuntimeError('Extract not found. You can use download=True to download it.')
        # Take the specified number of samples
        with open(file_path, 'rb') as f:
            extract = json.load(f)[:num_samples]

        return extract

    def _download_images(self, extract, threads=None):
        """ Download images from a specific extract

        Args:
            extract (list<dict>): image extract to download
            threads (int, optional): number of threads used for parallel downloading
        """

        self._images.mkdir(parents=True, exist_ok=True)
        # Prepare URL and filenames for multi-processing
        entries = [(s['url'], s['name']) for s in extract
                   if not self._images.joinpath(s['name']).is_file()]
        # Use multiple threads to speed up download
        if len(entries) > 0:
            download_urls(entries, self._images, threads=threads)

    def _verify_samples(self, extract):
        """ Download images from a specific extract

        Args:
            extract (list<dict>): list of samples
        Returns:
            valid_samples (list<dict>): list of valid samples
        """

        valid_samples = []
        dl_issues, target_issues = 0, 0
        # Verify samples in extract
        for sample in extract:

            is_ok = True
            # Verify image
            if not self._images.joinpath(sample['name']).is_file():
                dl_issues += 1
                is_ok = False

            # Verify targets
            if self.class_to_idx.get(sample['target']) is None:
                target_issues += 1
                is_ok = False

            if is_ok:
                valid_samples.append(sample)

        # HTTP errors
        if dl_issues == len(extract):
            raise RuntimeError('Images not found. You can use download=True to download them.')
        elif dl_issues > 0:
            warnings.warn(f'{dl_issues}/{len(extract)} sample images are not present on disk. '
                          'Please retry downloading later.')
        # Extract errors
        if target_issues > 0:
            warnings.warn(f'{target_issues}/{len(extract)} samples have corrupted targets.')

        return valid_samples

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
