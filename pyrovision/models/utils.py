# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

# Based on https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
# and https://github.com/fastai/fastai/blob/master/fastai/torch_core.py


import torch.nn as nn
from ..nn import AdaptiveConcatPool2d

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def init_module(m, init=nn.init.kaiming_normal_):
    """Initialize learnable parameters of a given module

    Args:
        m (torch.nn.Module): module to initialize
        init (callable, optional): inplace initializer function
    """

    # Apply init to learnable weights
    if hasattr(m, 'weight') and m.weight.requires_grad:
        init(m.weight)

    # Set learnable biases to 0.
    if hasattr(m, 'bias') and m.bias.requires_grad and hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.)


def head_stack(in_features, out_features, bn=True, p=0., actn=None):
    """Stacks batch norm, dropout and fully connected layers together

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        bn (bool, optional): should batchnorm be added
        p (float, optional): dropout probability
        actn (callable, optional): activation function
    Returns:
        torch.nn.Module: classifier head
    """
    layers = [nn.BatchNorm1d(in_features)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(in_features, out_features))
    if actn is not None:
        layers.append(actn)
    return layers


def create_head(in_features, num_classes, lin_features=512, dropout_prob=0.5,
                bn_final=False, concat_pool=True):
    """Instantiate a classifier head

    Args:
        in_features (int): number of input features
        num_classes (int): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers
        dropout_prob (float, optional): dropout probability
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by AdaptiveConcatPool2d
    Returns:
        torch.nn.Module: classifier head
    """
    # Pooling
    if concat_pool:
        pool = AdaptiveConcatPool2d((1, 1))
        in_features *= 2
    else:
        pool = nn.AdaptiveAvgPool2d((1, 1))

    # Nodes' layout
    if isinstance(lin_features, int):
        lin_features = [in_features, lin_features, num_classes]
    elif isinstance(lin_features, list):
        lin_features = [in_features] + lin_features + [num_classes]
    else:
        raise TypeError('expected argument lin_features to be of type int or list.')

    # Add half dropout probabilities for penultimate FC
    dropout_prob = [dropout_prob]
    if len(dropout_prob) == 1:
        dropout_prob = [dropout_prob[0] / 2] * (len(lin_features) - 2) + dropout_prob
    # ReLU activations except last FC
    activations = [nn.ReLU(inplace=True)] * (len(lin_features) - 2) + [None]

    # Flatten pooled feature maps
    layers = [pool, nn.Flatten()]
    for in_feats, out_feats, prob, activation in zip(lin_features[:-1], lin_features[1:], dropout_prob, activations):
        layers.extend(head_stack(in_feats, out_feats, True, prob, activation))
    # Final batch norm
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_features[-1], momentum=0.01))

    return nn.Sequential(*layers)


def create_body(model, cut):
    """Extracts the convolutional features from a model

    Args:
        model (torch.nn.Module): model
        cut (int): index of the first non-convolutional layer
    Returns:
        torch.nn.Module: model convolutional layerd
    """

    return nn.Sequential(*list(model.children())[:cut])


def cnn_model(base_model, cut, nb_features=None, num_classes=None, lin_features=512,
              dropout_prob=0.5, custom_head=None, bn_final=False, concat_pool=True,
              init=nn.init.kaiming_normal_):
    """Create a model with standard high-level structure as a torch.nn.Sequential

    Args:
        base_model (torch.nn.Module): base model
        cut (int): index of the first non-convolutional layer
        nb_features (int): number of convolutional features
        num_classes (int): number of output classes
        lin_features (Union[int, list<int>], optional): number of nodes in intermediate layers
        dropout_prob (float, optional): dropout probability
        custom_head (torch.nn.Module, optional): replacement for model's head
        bn_final (bool, optional): should a batch norm be added after the last layer
        concat_pool (bool, optional): should pooling be replaced by AdaptiveConcatPool2d
        init (callable, optional): initializer to use for model's head
    Returns:
        torch.nn.Module: instantiated model
    """

    body = create_body(base_model, cut)
    if custom_head is None:
        # Number of features
        if not (isinstance(nb_features, int) and isinstance(num_classes, int)):
            raise ValueError('nb_features & num_classes need to be specified when custom_head is None')
        head = create_head(nb_features, num_classes, lin_features, dropout_prob, bn_final, concat_pool)
    else:
        head = custom_head

    # Init all non-BN layers
    if init:
        for m in head:
            if (not isinstance(m, bn_types)):
                init_module(m, init)

    return nn.Sequential(body, head)
