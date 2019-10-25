import unittest
import torch
from pyronear import nn

# Based on https://github.com/pytorch/pytorch/blob/master/test/test_nn.py


class NNTester(unittest.TestCase):

    def test_adaptive_pooling_input_size(self):
        for numel in (2,):
            for pool_type in ('Concat',):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * numel
                module = module_cls(output_size)

                input = torch.randn(output_size)
                self.assertRaises(ValueError, lambda: module(input))

    def test_adaptive_pooling_size_none(self):
        for numel in (2,):
            for pool_type in ('Concat',):
                cls_name = 'Adaptive{}Pool{}d'.format(pool_type, numel)
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * (numel - 1) + (None,)
                module = module_cls(output_size)

                input = torch.randn((4,) * (numel + 1))
                output = module(input)
                self.assertEqual(output.size(), (4,) + (4,) * (numel - 1) + (4,))


if __name__ == '__main__':
    unittest.main()
