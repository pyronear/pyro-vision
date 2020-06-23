# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest
import torch
from pyronear import nn

# Based on https://github.com/pytorch/pytorch/blob/master/test/test_nn.py


class NNTester(unittest.TestCase):

    def test_adaptive_pooling_input_size(self):
        for numel in (2,):
            for pool_type in ('Concat',):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                output_size = (2,) * numel
                module = nn.__dict__[cls_name](output_size)

                x = torch.randn(output_size)
                self.assertRaises(ValueError, lambda: module(x))

    def test_adaptive_pooling_size_none(self):
        for numel in (2,):
            for pool_type in ('Concat',):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                output_size = (2,) * (numel - 1) + (None,)
                module = nn.__dict__[cls_name](output_size)

                x = torch.randn((4,) * (numel + 1))
                output = module(x)
                self.assertEqual(output.size(), (4,) + (4,) * (numel - 1) + (4,))


if __name__ == '__main__':
    unittest.main()
