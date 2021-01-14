 # -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import torch
from pyrovision.models.utils import cnn_model
import holocron
from holocron.models.utils import load_pretrained_params


## Define Model
backbone = 'rexnet1_0x'
num_classes = 1
nb_features = 1280
cut = -2
url = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.0/rexnet1_0x_acp_2e017f83.pth"


def pyronear_model(pretrain=True, device='cpu'):
    """The purpose of this function is to return the latest Pyronear model."""
    # Get backbone
    base = holocron.models.__dict__[backbone](False, num_classes=num_classes)
    # Change head
    model = cnn_model(base, cut), nb_features=nb_features, num_classes=num_classes)
    # Load Weight
    if pretrain:
        load_pretrained_params(model, url)
    # Move to gpu
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda:0')

    return model
