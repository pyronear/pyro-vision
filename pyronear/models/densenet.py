#!usr/bin/python
# -*- coding: utf-8 -*-

import re
from torchvision.models.densenet import DenseNet, model_urls as imagenet_urls
from torchvision.models.utils import load_state_dict_from_url
from .utils import cnn_model

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {}

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
              progress=True, imagenet_pretrained=False, num_classes=1, **kwargs):
    r"""Instantiate a DenseNet model for image classification from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>

    Args:
        arch (str): densenet architecture
        growth_rate (int): growth rate of each dense block
        block_config (list<int>): number of blocks in each layer
        num_init_features (int): number of output channels in the first convolution
        pretrained (bool, optional): should pretrained parameters be loaded (OpenFire training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        **kwargs: optional arguments of torchvision.models.densenet.DenseNet

    Returns:
        model (torch.nn.Module): loaded model
    """

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
    model = cnn_model(base_model, cut=model_cut, nb_features=base_model.classifier.in_features,
                      num_classes=num_classes)

    # Parameter loading
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def densenet121(*args, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, *args, **kwargs)


def densenet161(*args, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, *args, **kwargs)


def densenet169(*args, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, *args, **kwargs)


def densenet201(*args, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, *args, **kwargs)
