# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict

from holocron.models.presets import IMAGENET
from holocron.models.utils import load_pretrained_params
from torchvision.models import mobilenetv3 as src

__all__ = ["mobilenet_v3_small", "mobilenet_v3_large"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_small": {
        **IMAGENET,
        "classes": ["Wildfire"],
        "input_shape": (3, 224, 224),
        "resize_mode": "squish",
        "url": "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/mobilenet_v3_small_224-619caf97.pth",
    },
    "mobilenet_v3_large": {
        **IMAGENET,
        "classes": ["Wildfire"],
        "input_shape": (3, 224, 224),
        "resize_mode": "squish",
        "url": "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/mobilenet_v3_large_224-33e1b104.pth",
    },
}


def _mobilenet_v3(
    arch_fn: Callable[[Any], src.MobileNetV3],
    arch: str,
    pretrained: bool,
    progress: bool,
    num_classes: int = 1,
    **kwargs: Any,
) -> src.MobileNetV3:

    # Build the model
    model = arch_fn(num_classes=num_classes, **kwargs)  # type: ignore[call-arg]
    model.default_cfg = default_cfgs[arch]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.MobileNetV3:
    """MobileNetV3 model from
    `"Searching for MobileNetV" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _mobilenet_v3(src.mobilenet_v3_small, "mobilenet_v3_small", pretrained, progress, **kwargs)


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> src.MobileNetV3:
    """MobileNetV3 model from
    `"Searching for MobileNetV" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    return _mobilenet_v3(src.mobilenet_v3_large, "mobilenet_v3_large", pretrained, progress, **kwargs)
