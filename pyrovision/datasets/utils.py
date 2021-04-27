# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import urllib.request
import hashlib
from tqdm import tqdm
import zipfile
from pathlib import Path


class DownloadProgressBar(tqdm):
    """Progress bar."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset_from_url(url, root):
    """Download datatset."""
    output_path = url.split('/')[-1]
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        # check integrity of downloaded file
        with open(output_path, 'rb') as f:
            sha_hash = hashlib.sha256(f.read()).hexdigest()
        if sha_hash[:8] != url.split('-')[-1].split('.')[0]:
            raise RuntimeError("File not found or corrupted.")
        print("\nUnziping ...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        print('Done!')
