import unittest
from pathlib import Path
import json
from PIL.Image import Image
from torchvision.datasets import VisionDataset

from pyronear import datasets


class TestCollectEnv(unittest.TestCase):
    def test_downloadurl(self):
        # URL
        wrong_url1 = 'url'
        wrong_url2 = 0
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'

        # Root
        wrong_root = 0
        root = '/tmp'

        # URL error cases
        self.assertRaises(ValueError, datasets.utils.download_url, wrong_url1, root, verbose=False)
        self.assertRaises(TypeError, datasets.utils.download_url, wrong_url2, root, verbose=False)

        # Root error cases
        self.assertRaises(TypeError, datasets.utils.download_url, url, wrong_root, verbose=False)

        # Working case
        datasets.utils.download_url(url, root, verbose=True)
        self.assertTrue(Path(root, url.rpartition('/')[-1]).is_file())

    def test_openfire(self):

        root = '/tmp'

        # Working case
        # Check inherited properties
        dataset = datasets.OpenFire(root=root, train=True, download=True)
        self.assertTrue(isinstance(dataset, VisionDataset))

        # Check against number of samples in extract
        datasets.utils.download_url(dataset.url, root, filename='extract.json', verbose=False)
        with open(Path(root, 'extract.json'), 'rb') as f:
            extract = json.load(f)
        self.assertEqual(len(dataset), len(extract))

        # Check integrity of samples
        img, target = dataset[0]
        self.assertTrue(isinstance(img, Image))
        self.assertTrue(isinstance(target, int))
        self.assertEqual(dataset.class_to_idx[extract[0]['target']], target)


if __name__ == '__main__':
    unittest.main()