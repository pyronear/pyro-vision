# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import multiprocessing as mp
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

__all__ = ["download_url", "download_urls"]


def url_retrieve(url: str, outfile: Path, timeout: int = 4) -> None:
    """Download the content of an URL request to a specified location

    Args:
        url (str): URL to request
        outfile (pathlib.Path): path of the file where the response will be saved
        timeout (float, optional): number of seconds before the request times out
    """

    response = requests.get(url, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise requests.exceptions.ConnectionError(f"Error code {response.status_code} - could not download {url}")

    outfile.write_bytes(response.content)


def get_fname(url: str, default_extension: str = "jpg", max_base_length: Optional[int] = None) -> str:
    """Find extension of file located by URL

    Args:
        url (str): URL of the file
        default_extension (str, optional): default extension
        max_base_length (int, optional): max base filename's length

    Returns:
        str: file name
    """

    name_split = urlparse(url).path.rpartition("/")[-1].split("?")[0].split("&")[0].split(";")[0].split(".")
    # Check if viable extension
    if len(name_split) > 1 and all(c.isalpha() or c.isdigit() for c in name_split[-1].lower()):
        base, extension = ".".join(name_split[:-1]), name_split[-1].lower()
    # Fallback on default extension
    else:
        base, extension = name_split[-1], default_extension
    # Check base length
    if isinstance(max_base_length, int) and len(base) > max_base_length:
        base = base[:max_base_length]

    return f"{base}.{extension}"


def download_url(
    url: str,
    root: Path,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    timeout: int = 4,
    retries: int = 4,
    verbose: bool = False,
    silent: bool = False,
) -> None:
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
        raise TypeError("expected argument url to be of type <str>")

    # Root folder
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = get_fname(url)

    fpath = root.joinpath(filename)

    # Download file
    if check_integrity(fpath, md5):
        if verbose:
            print(f"Using downloaded and verified file: {fpath}")
    else:
        success = False
        # Allow multiple retries
        for idx in range(retries + 1):
            try:
                url_retrieve(url, fpath, timeout)
                success = True
            except Exception as e:
                # Try switching to http
                if url.startswith("https"):
                    try:
                        url_retrieve(url.replace("https:", "http:"), fpath, timeout)
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


def parallel(
    func: Callable[[Any], Any],
    arr: Sequence[Any],
    num_threads: Optional[int] = None,
    leave: bool = False,
) -> Optional[Sequence[Any]]:
    """Download a file accessible via URL with mutiple retries

    Args:
        func (callable): function to be executed on multiple workers
        arr (iterable): function argument's values
        num_threads (int, optional): number of workers to be used for multiprocessing
        leave (bool, optional): whether traces of progressbar should be kept upon termination

    Returns:
        list: list of function's results
    """

    if num_threads is None:
        num_threads = min(16, mp.cpu_count())
    if num_threads < 2:
        results = [func(arg) for arg in tqdm(arr, total=len(arr), leave=leave)]
    else:
        with ThreadPool(num_threads) as tp:
            results = list(tqdm(tp.imap_unordered(func, arr), total=len(arr)))
    if any([o is not None for o in results]):
        return results

    return None


def download_urls(
    entries: List[Tuple[str, str]],
    root: Path,
    timeout: int = 4,
    retries: int = 4,
    num_threads: Optional[int] = None,
    silent: bool = True,
) -> None:
    """Download multiple URLs a file accessible via URL with mutiple retries

    Args:
        entries (list<str, str>): URL and destination filen
        root (pathlib.Path): folder where the files will be saved in
        timeout (float, optional): number of seconds before the request times out
        retries (int, optional): number of additional allowed download attempts
        num_threads (int, optional): number of threads to be used for multiprocessing
        silent (bool, optional): whether Exception should be raised upon download failure
    """

    parallel(
        partial(download_url, root=root, timeout=timeout, retries=retries, silent=silent),
        entries,
        num_threads=num_threads,
    )
