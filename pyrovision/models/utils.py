# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# Borrowed from https://github.com/frgfm/Holocron/blob/main/holocron/models/utils.py

import json
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from pyrovision import models

__all__ = ["model_from_hf_hub"]


def model_from_hf_hub(repo_id: str, **kwargs: Any) -> nn.Module:
    """Instantiate & load a pretrained model from HF hub.

    Args:
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download`
    Returns:
        Model loaded with the checkpoint
    """

    # Get the config
    with open(hf_hub_download(repo_id, filename="config.json", **kwargs), "rb") as f:
        cfg = json.load(f)

    model = models.__dict__[cfg["arch"]](num_classes=len(cfg["classes"]), pretrained=False)
    # Patch the config
    model.default_cfg.update(cfg)

    # Load the checkpoint
    state_dict = torch.load(hf_hub_download(repo_id, filename="pytorch_model.bin", **kwargs), map_location="cpu")
    model.load_state_dict(state_dict)

    return model
