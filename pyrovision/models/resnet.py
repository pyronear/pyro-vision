# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict

from holocron.models.presets import IMAGENET
from holocron.models.utils import load_pretrained_params
from torchvision.models import resnet as src

__all__ = ["resnet18", "resnet34"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        **IMAGENET,
        "classes": ["Wildfire"],
        "input_shape": (3, 224, 224),
        "url": "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/resnet18_224-aa7b3886.pth",
    },
    "resnet34": {
        **IMAGENET,
        "classes": ["Wildfire"],
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


def _resnet(
    arch_fn: Callable[[Any], src.ResNet],
    arch: str,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> src.ResNet:
    # Build the model
    model = arch_fn(**kwargs)  # type: ignore[call-arg]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.ResNet:
    """ResNet-18 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _resnet(src.resnet18, "resnet18", pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.ResNet:
    """ResNet-34 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _resnet(src.resnet34, "resnet34", pretrained, progress, **kwargs)
