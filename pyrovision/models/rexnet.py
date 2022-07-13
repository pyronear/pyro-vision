# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict

from holocron.models import rexnet as src
from holocron.models.presets import IMAGENET
from holocron.models.utils import load_pretrained_params

__all__ = ["rexnet1_0x", "rexnet1_3x", "rexnet1_5x"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "rexnet1_0x": {
        **IMAGENET,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "rexnet1_3x": {
        **IMAGENET,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "rexnet1_5x": {
        **IMAGENET,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


def _rexnet(
    arch_fn: Callable[[Any], src.ReXNet],
    arch: str,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> src.ReXNet:
    # Build the model
    model = arch_fn(**kwargs)  # type: ignore[call-arg]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def rexnet1_0x(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.ReXNet:
    """ReXNet-1.0x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _rexnet(src.rexnet1_0x, "rexnet1_0x", pretrained, progress, **kwargs)


def rexnet1_3x(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.ReXNet:
    """ReXNet-1.3x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _rexnet(src.rexnet1_0x, "rexnet1_0x", pretrained, progress, **kwargs)


def rexnet1_5x(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.ReXNet:
    """ReXNet-1.5x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _rexnet(src.rexnet1_0x, "rexnet1_0x", pretrained, progress, **kwargs)
