#!usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
from six.moves import urllib
from torchvision.datasets.utils import check_integrity, gen_bar_updater


def download_url(url, root, filename=None, md5=None, verbose=True):
    """Download a file from a url and place it in root. Based on torchvision.datasets.utils.download_url.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        verbose (bool, optional): Should download progress and verbose be displayed in console
    """

    if not isinstance(url, str):
        raise TypeError('expected argument url to be of type <str>')

    root = Path(root).expanduser()
    if not filename:
        filename = url.rpartition('/')[-1]
    fpath = root.joinpath(filename)

    root.mkdir(parents=True, exist_ok=True)

    # downloads file
    if check_integrity(fpath, md5):
        if verbose:
            print(f'Using downloaded and verified file: {fpath}')
    else:
        try:
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater() if verbose else None)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                if verbose:
                    print('Failed download. Trying https -> http instead.'
                          f' Downloading {url} to {fpath}')
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater() if verbose else None)
            else:
                raise e
