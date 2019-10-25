#!usr/bin/python
# -*- coding: utf-8 -*-


from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, model_urls as imagenet_urls
from torchvision.models.utils import load_state_dict_from_url
from .utils import cnn_model


model_urls = {
    'resnet18': 'https://srv-file7.gofile.io/download/rG1vDJ/resnet18-binary-classification.pth',
    'resnet34': 'https://srv-file7.gofile.io/download/W3gb2q/resnet34-binary-classification.pth'
}


def _resnet(arch, block, layers, pretrained=False, progress=True,
            imagenet_pretrained=False, num_classes=1, **kwargs):
    r"""Instantiate a ResNet model for image classification from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        arch (str): resnet architecture
        block (torch.nn.Module): block used as bottleneck
        layers (list<int>): number of blocks in each layer
        pretrained (bool, optional): should pretrained parameters be loaded (OpenFire training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        imagenet_pretrained (bool, optional): should pretrained parameters be loaded on conv layers (ImageNet training)
        num_classes (int, optional): number of output classes
        **kwargs: optional arguments of torchvision.models.resnet.ResNet

    Returns:
        model (torch.nn.Module): loaded model
    """

    # Model creation
    base_model = ResNet(block, layers, num_classes=num_classes, **kwargs)
    # Imagenet pretraining
    if imagenet_pretrained:
        if pretrained:
            raise ValueError('imagenet_pretrained cannot be set to True if pretrained=True')
        state_dict = load_state_dict_from_url(imagenet_urls[arch],
                                              progress=progress)
        # Remove FC params from dict
        for key in ('fc.weight', 'fc.bias'):
            state_dict.pop(key, None)
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if any(unexpected) or any(not elt.startswith('fc.') for elt in missing):
            raise KeyError(f"Missing parameters: {missing}\nUnexpected parameters: {unexpected}")

    # Cut at last conv layers
    model = cnn_model(base_model, cut=-2, nb_features=base_model.fc.in_features,
                      num_classes=num_classes)

    # Parameter loading
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def resnet18(*args, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], *args, **kwargs)


def resnet34(*args, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], *args, **kwargs)


def resnet50(*args, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], *args, **kwargs)


def resnet101(*args, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], *args, **kwargs)


def resnet152(*args, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], *args, **kwargs)
