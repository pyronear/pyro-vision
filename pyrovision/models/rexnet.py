# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from holocron.models.rexnet import ReXNet
from typing import Dict, Any
from pyrovision.models.utils import cnn_model
from holocron.models.utils import load_pretrained_params

__all__ = ['rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x', 'rexnet2_2x']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'rexnet1_0x': {'width_mult': 1.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_0x_224-ab7b9733.pth'},
    'rexnet1_3x': {'width_mult': 1.3, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_3x_224-95479104.pth'},
    'rexnet1_5x': {'width_mult': 1.5, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_5x_224-c42a16ac.pth'},
    'rexnet2_0x': {'width_mult': 2.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet2_0x_224-c8802402.pth'},
    'rexnet2_2x': {'width_mult': 2.2, 'depth_mult': 1.0,
                   'url': None},
}

model_urls = {
    'rexnet1_0x': "https://github.com/pyronear/pyro-vision/releases/download/v0.1.0/rexnet1_0x_acp_2e017f83.pth",
    'rexnet1_3x': None,
    'rexnet1_5x': None,
    'rexnet2_0x': None,
    'rexnet2_2x': None
}

model_cut = -2


def _rexnet(arch, pretrained=False, progress=True,
            imagenet_pretrained=False, num_classes=1, lin_features=512,
            dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):

    # Model creation
    base_model = ReXNet(default_cfgs[arch]['width_mult'], default_cfgs[arch]['depth_mult'], **kwargs)

    # Imagenet pretraining
    if imagenet_pretrained:
        if pretrained:
            raise ValueError('imagenet_pretrained cannot be set to True if pretrained=True')

        load_pretrained_params(base_model, default_cfgs[arch]['url'], progress)

    # Cut at last conv layers
    model = cnn_model(base_model, model_cut, base_model.head[1].in_features, num_classes,
                      lin_features, dropout_prob, bn_final=bn_final, concat_pool=concat_pool)

    # Parameter loading
    if pretrained:
        load_pretrained_params(model, model_urls[arch], progress=progress)

    return model


def rexnet1_0x(pretrained=False, progress=True, imagenet_pretrained=False, num_classes=1,
               lin_features=512, dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):

    r"""ReXNet-1.0x from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool, optional): should pretrained parameters be loaded (Pyronear training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.resnet.ResNet`
    """
    return _rexnet('rexnet1_0x', pretrained, progress, imagenet_pretrained, num_classes,
                   lin_features, dropout_prob, bn_final, concat_pool, **kwargs)


def rexnet1_3x(pretrained=False, progress=True, **kwargs):
    r"""ReXNet-1.3x from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool, optional): should pretrained parameters be loaded (Pyronear training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.resnet.ResNet`
    """
    return _rexnet('rexnet1_3x', pretrained, progress, **kwargs)


def rexnet1_5x(pretrained=False, progress=True, **kwargs):
    r"""ReXNet-1.5x from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool, optional): should pretrained parameters be loaded (Pyronear training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.resnet.ResNet`
    """
    return _rexnet('rexnet1_5x', pretrained, progress, **kwargs)


def rexnet2_0x(pretrained=False, progress=True, **kwargs):
    r"""ReXNet-2.0x from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool, optional): should pretrained parameters be loaded (Pyronear training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.resnet.ResNet`
    """
    return _rexnet('rexnet2_0x', pretrained, progress, **kwargs)


def rexnet2_2x(pretrained=False, progress=True, **kwargs):
    r"""ReXNet-2.2x from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool, optional): should pretrained parameters be loaded (Pyronear training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.resnet.ResNet`
    """
    return _rexnet('rexnet2_2x', pretrained, progress, **kwargs)
