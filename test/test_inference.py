# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest
from pyrovision.inference.pyronear_predictor import PyronearPredictor
from PIL import Image
from pathlib import Path


class PyronearPredictorTester(unittest.TestCase):

    def setUp(self):
        self.config = Path(__file__).parent.parent / 'pyrovision/inference/inference.cfg'
        self.testImage = Path(__file__).parent / 'fixtures/wildfire_example.jpg'
        print(self.config, self.testImage)

    def test_pyronear_predictor(self):
        # Create Pyronear Predictor
        pyronearPredictor = PyronearPredictor(self.config)

        # Load Image
        im = Image.open(self.testImage)

        # Make Prediction
        pred = pyronearPredictor.predict(im)

        self.assertGreater(pred, 0.5)


if __name__ == '__main__':
    unittest.main()
