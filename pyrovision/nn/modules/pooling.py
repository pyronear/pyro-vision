# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

# Based on https://github.com/fastai/fastai/blob/master/fastai/layers.py


import torch.nn as nn
from .. import functional as F


class AdaptiveConcatPool2d(nn.Module):
    r"""Applies both a 2D adaptive max pooling and a 2D adaptive average pooling over an input
    signal composed of several input planes and concatenates them.
    The output is of size H x W, for any input size.
    The number of output features is equal to twice the number of input planes.

    Args:
        output_size (Union[int, tuple<int>]): the target output size of the image of the form H x W.
            Can be a tuple (H, W) or a single H for a square image H x H.
            H and W can be either a ``int``, or ``None`` which means the size will
            be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveConcatPool2d((5,7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveConcatPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveConcatPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
    """
    __constants__ = ['output_size', 'return_indices']

    def __init__(self, output_size):
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_concat_pool2d(x, self.output_size)

    def extra_repr(self):
        return 'output_size={}'.format(self.output_size)
