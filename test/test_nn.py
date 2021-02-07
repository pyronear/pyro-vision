# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import unittest
import torch
from pyrovision import nn

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
