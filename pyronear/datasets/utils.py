#!usr/bin/python
# -*- coding: utf-8 -*-

import os
from torchvision.datasets.utils import makedir_exist_ok, check_integrity, gen_bar_updater


def download_url(url, root, filename=None, md5=None, verbose=True):
    """Download a file from a url and place it in root. Based on torchvision.datasets.utils.download_url.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        verbose (bool, optional): Should download progress and verbose be displayed in console
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

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
