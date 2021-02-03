# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import unittest
import tempfile
from pathlib import Path
import requests

from pyrovision.datasets import utils


class DatasetsUtilsTester(unittest.TestCase):
    def test_downloadurl(self):
        # Valid input
        url = 'https://arxiv.org/pdf/1910.02940.pdf'

        with tempfile.TemporaryDirectory() as root:
            # URL error cases
            self.assertRaises(requests.exceptions.MissingSchema, utils.download_url,
                              'url', root, verbose=False)
            self.assertRaises(requests.exceptions.ConnectionError, utils.download_url,
                              'https://url', root, verbose=False)
            self.assertRaises(TypeError, utils.download_url, 0, root, verbose=False)

            # Root error cases
            self.assertRaises(TypeError, utils.download_url, url, 0, verbose=False)

            # Working case
            utils.download_url(url, root, verbose=True)
            self.assertTrue(Path(root, url.rpartition('/')[-1]).is_file())

    def test_downloadurls(self):
        # Valid input
        urls = ['https://arxiv.org/pdf/1910.01108.pdf', 'https://arxiv.org/pdf/1810.04805.pdf',
                'https://arxiv.org/pdf/1905.11946.pdf', 'https://arxiv.org/pdf/1910.01271.pdf']

        with tempfile.TemporaryDirectory() as root:
            # URL error cases
            self.assertRaises(requests.exceptions.MissingSchema, utils.download_urls,
                              ['url'] * 4, root, silent=False)
            self.assertRaises(requests.exceptions.ConnectionError, utils.download_urls,
                              ['https://url'] * 4, root, silent=False)
            self.assertRaises(TypeError, utils.download_url, [0] * 4, root, silent=False)

            # Working case
            utils.download_urls(urls, root, silent=False)
            self.assertTrue(all(Path(root, url.rpartition('/')[-1]).is_file() for url in urls))


if __name__ == '__main__':
    unittest.main()
