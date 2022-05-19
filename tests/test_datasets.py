# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import json
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import requests
from PIL.Image import Image
from torchvision.datasets import VisionDataset

from pyrovision import datasets


def generate_wildfire_dataset_fixture():
    random.seed(42)
    df = pd.DataFrame(columns=['imgFile', 'fire_id', 'fire'])
    for i in range(974):
        df = df.append({'imgFile': str(i).zfill(4) + '.jpg', 'fire_id': float(random.randint(1, 100)),
                        'fire': float(random.randint(0, 1))}, ignore_index=True)

    return df


def generate_wildfire_subsampler_dataset_fixture():
    df = pd.DataFrame(columns=['exploitable', 'fire', 'sequence', 'clf_confidence',
                               'loc_confidence', 'x', 'y', 't', 'stateStart',
                               'stateEnd', 'imgFile', 'fire_id', 'fBase'])
    for b in range(10):
        x = random.uniform(200, 500)
        y = random.uniform(200, 500)
        t = random.uniform(0, 100)
        start = random.randint(0, 200)
        end = random.randint(start + 11, 400)
        base = str(b) + '.mp4'
        imgsNb = random.sample(range(start, end), 10)
        imgsNb.sort()
        imgs = [str(b) + '_frame' + str(i) + '.png' for i in imgsNb]
        fire_id = float(random.randint(1, 100))
        fire = float(random.randint(0, 1))
        for i in range(10):
            df = df.append({'exploitable': True, 'fire': fire, 'sequence': 0,
                            'clf_confidence': 0, 'loc_confidence': 0, 'x': x, 'y': y, 't': t, 'stateStart': start,
                            'stateEnd': end, 'imgFile': imgs[i], 'fire_id': fire_id,
                            'fBase': base}, ignore_index=True)

    return df


def get_wildfire_image():

    #download image
    url = 'https://media.springernature.com/w580h326/nature-cms/uploads/collections/' \
          'Wildfire-and-ecosystems-Hero-d62e7fbbf36ce6915d4e3efef069ee0e.jpg'
    response = requests.get(url)
    # save image
    file = open("test//0003.jpg", "wb")
    file.write(response.content)
    file.close()


class OpenFireTester(unittest.TestCase):
    def test_openfire(self):
        num_samples = 200

        # Test img_folder argument: wrong type and default (None)
        with tempfile.TemporaryDirectory() as root:
            self.assertRaises(TypeError, datasets.OpenFire, root, download=True, img_folder=1)
            ds = datasets.OpenFire(root=root, download=True, num_samples=num_samples,
                                   img_folder=None)
            self.assertIsInstance(ds.img_folder, Path)

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as img_folder:

            # Working case
            # Test img_folder as Path and str
            train_set = datasets.OpenFire(root=root, train=True, download=True, num_samples=num_samples,
                                          img_folder=Path(img_folder))
            test_set = datasets.OpenFire(root=root, train=False, download=True, num_samples=num_samples,
                                         img_folder=img_folder)
            # Check inherited properties
            self.assertIsInstance(train_set, VisionDataset)

            # Assert valid extensions of every image
            self.assertTrue(all(sample['name'].rpartition('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']
                                for sample in train_set.data))
            self.assertTrue(all(sample['name'].rpartition('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']
                                for sample in test_set.data))

            # Check against number of samples in extract (limit to num_samples)
            datasets.utils.download_url(train_set.url, root, filename='extract.json', verbose=False)
            with open(Path(root).joinpath('extract.json'), 'rb') as f:
                extract = json.load(f)[:num_samples]
            # Test if not more than 15 downloads failed.
            # Change to assertEqual when download issues are resolved
            self.assertAlmostEqual(len(train_set) + len(test_set), len(extract), delta=30)

            # Check integrity of samples
            img, target = train_set[0]
            self.assertIsInstance(img, Image)
            self.assertIsInstance(target, int)
            self.assertEqual(train_set.class_to_idx[extract[0]['target']], target)

            # Check train/test split
            self.assertIsInstance(train_set, VisionDataset)
            # Check unicity of sample across all splits
            train_paths = [sample['name'] for sample in train_set.data]
            self.assertTrue(all(sample['name'] not in train_paths for sample in test_set.data))


if __name__ == '__main__':
    unittest.main()
