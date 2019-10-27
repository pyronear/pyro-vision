#!usr/bin/python
# -*- coding: utf-8 -*-

from torchvision.models.mobilenet import MobileNetV2, model_urls as imagenet_urls
from torchvision.models.utils import load_state_dict_from_url
from .utils import cnn_model

__all__ = ['mobilenet_v2']


model_urls = {}

model_cut = -1


def mobilenet_v2(pretrained=False, progress=True, imagenet_pretrained=False,
                 num_classes=1, **kwargs):
    r"""Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        **kwargs: optional arguments of torchvision.models.mobilenet.MobileNetV2

    Returns:
        model (torch.nn.Module): loaded model
    """

    # Model creation
    base_model = MobileNetV2(num_classes=num_classes, **kwargs)
    # Imagenet pretraining
    if imagenet_pretrained:
        if pretrained:
            raise ValueError('imagenet_pretrained cannot be set to True if pretrained=True')
        state_dict = load_state_dict_from_url(imagenet_urls['mobilenet_v2'],
                                              progress=progress)
        # Remove FC params from dict
        for key in ('classifier.1.weight', 'classifier.1.bias'):
            state_dict.pop(key, None)
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if any(unexpected) or any(not elt.startswith('classifier.') for elt in missing):
            raise KeyError(f"Missing parameters: {missing}\nUnexpected parameters: {unexpected}")

    # Cut at last conv layers
    model = cnn_model(base_model, cut=model_cut, nb_features=base_model.classifier[1].in_features,
                      num_classes=num_classes)

    # Parameter loading
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model
