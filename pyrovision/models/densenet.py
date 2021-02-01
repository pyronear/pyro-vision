# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import re
from torchvision.models.densenet import DenseNet, model_urls as imagenet_urls
from torchvision.models.utils import load_state_dict_from_url
from .utils import cnn_model

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://srv-file7.gofile.io/download/XqHLBB/densenet121-binary-classification.pth'
}

model_cut = -1


def _update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained=False,
              progress=True, imagenet_pretrained=False, num_classes=1, lin_features=512,
              dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):

    # Model creation
    base_model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
    # Imagenet pretraining
    if imagenet_pretrained:
        if pretrained:
            raise ValueError('imagenet_pretrained cannot be set to True if pretrained=True')
        state_dict = load_state_dict_from_url(imagenet_urls[arch],
                                              progress=progress)
        state_dict = _update_state_dict(state_dict)
        # Remove FC params from dict
        for key in ('classifier.weight', 'classifier.bias'):
            state_dict.pop(key, None)
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if any(unexpected) or any(not elt.startswith('classifier.') for elt in missing):
            raise KeyError(f"Missing parameters: {missing}\nUnexpected parameters: {unexpected}")

    # Cut at last conv layers
    model = cnn_model(base_model, model_cut, base_model.classifier.in_features, num_classes,
                      lin_features, dropout_prob, bn_final=bn_final, concat_pool=concat_pool)

    # Parameter loading
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def densenet121(pretrained=False, progress=True, imagenet_pretrained=False, num_classes=1,
                lin_features=512, dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.densenet.DenseNet`
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     imagenet_pretrained, num_classes, lin_features, dropout_prob,
                     bn_final, concat_pool, **kwargs)


def densenet161(pretrained=False, progress=True, imagenet_pretrained=False, num_classes=1,
                lin_features=512, dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.densenet.DenseNet`
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     imagenet_pretrained, num_classes, lin_features, dropout_prob,
                     bn_final, concat_pool, **kwargs)


def densenet169(pretrained=False, progress=True, imagenet_pretrained=False, num_classes=1,
                lin_features=512, dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.densenet.DenseNet`
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     imagenet_pretrained, num_classes, lin_features, dropout_prob,
                     bn_final, concat_pool, **kwargs)


def densenet201(pretrained=False, progress=True, imagenet_pretrained=False, num_classes=1,
                lin_features=512, dropout_prob=0.5, bn_final=False, concat_pool=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers of model's head
        dropout_prob (float, optional): dropout probability of head FC layers
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by :mod:`pyronear.nn.AdaptiveConcatPool2d`
        **kwargs: optional arguments of :mod:`torchvision.models.densenet.DenseNet`
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     imagenet_pretrained, num_classes, lin_features, dropout_prob,
                     bn_final, concat_pool, **kwargs)
