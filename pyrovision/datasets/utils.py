# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import requests
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from pathlib import Path
from functools import partial
from tqdm import tqdm
from urllib.parse import urlparse
from torchvision.datasets.utils import check_integrity

__all__ = ['download_url', 'download_urls']


def url_retrieve(url, outfile, timeout=4):
    """Download the content of an URL request to a specified location

    Args:
        url (str): URL to request
        outfile (pathlib.Path): path of the file where the response will be saved
        timeout (float, optional): number of seconds before the request times out
    """

    response = requests.get(url, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise requests.exceptions.ConnectionError(f'Error code {response.status_code} - could not download {url}')

    outfile.write_bytes(response.content)


def get_fname(url, default_extension='jpg', max_base_length=50):
    """Find extension of file located by URL

    Args:
        url (str): URL of the file
        default_extension (str, optional): default extension
        max_base_length (int, optional): max base filename's length

    Returns:
        str: file name
    """

    name_split = urlparse(url).path.rpartition('/')[-1].split('.')
    # Check if viable extension
    if len(name_split) > 1 and all(c.isalpha() for c in name_split[-1].lower()):
        base, extension = '.'.join(name_split[:-1]), name_split[-1].lower()
    # Fallback on default extension
    else:
        base, extension = name_split[-1], default_extension
    # Check base length
    if len(base) > max_base_length:
        base = base[:max_base_length]

    return f"{base}.{extension}"


def download_url(url, root, filename=None, md5=None, timeout=4,
                 retries=4, verbose=False, silent=False):
    """Download a file accessible via URL with mutiple retries

    Args:
        url (str or tuple<str, str>): URL to request
        root (pathlib.Path): folder where the file will be saved in
        filename (str, optional): name of the output file
        md5 (str, optional): md5 for integrity verification
        timeout (float, optional): number of seconds before the request times out
        retries (int, optional): number of additional allowed download attempts
        verbose (bool, optional): whether status can be displayed in console
        silent (bool, optional): whether Exception should be raised upon download failure
    """

    if isinstance(url, tuple):
        url, filename = url

    if not isinstance(url, str):
        raise TypeError('expected argument url to be of type <str>')

    # Root folder
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = get_fname(url)

    fpath = root.joinpath(filename)

    # Download file
    if check_integrity(fpath, md5):
        if verbose:
            print(f'Using downloaded and verified file: {fpath}')
    else:
        success = False
        # Allow multiple retries
        for idx in range(retries + 1):
            try:
                url_retrieve(url, fpath, timeout)
                success = True
            except Exception as e:
                # Try switching to http
                if url.startswith('https'):
                    try:
                        url_retrieve(url.replace('https:', 'http:'), fpath, timeout)
                        success = True
                    except Exception:
                        success = False
                # Handle exception
                if not success and (idx == retries):
                    if not silent:
                        raise e
                    elif verbose:
                        print(e)
            if success:
                break


def parallel(func, arr, threads=None, leave=False):
    """Download a file accessible via URL with mutiple retries

    Args:
        func (callable): function to be executed on multiple workers
        arr (iterable): function argument's values
        threads (int, optional): number of workers to be used for multiprocessing
        leave (bool, optional): whether traces of progressbar should be kept upon termination

    Returns:
        list: list of function's results
    """

    if threads is None:
        threads = min(16, mp.cpu_count())
    if threads < 2:
        results = [func(arg) for arg in tqdm(arr, total=len(arr), leave=leave)]
    else:
        with ThreadPool(threads) as tp:
            results = list(tqdm(tp.imap_unordered(func, arr), total=len(arr)))
    if any([o is not None for o in results]):
        return results


def download_urls(entries, root, timeout=4, retries=4, threads=None, silent=True):
    """Download multiple URLs a file accessible via URL with mutiple retries

    Args:
        entries (list<str, str>): URL and destination filen
        root (pathlib.Path): folder where the files will be saved in
        timeout (float, optional): number of seconds before the request times out
        retries (int, optional): number of additional allowed download attempts
        threads (int, optional): number of threads to be used for multiprocessing
        silent (bool, optional): whether Exception should be raised upon download failure
    """

    parallel(partial(download_url, root=root, timeout=timeout, retries=retries, silent=silent),
             entries, threads=threads)
