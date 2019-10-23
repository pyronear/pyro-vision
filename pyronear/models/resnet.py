#!usr/bin/python
# -*- coding: utf-8 -*-


from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.utils import load_state_dict_from_url


resnet_layers = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}

model_urls = {
	18: 'https://srv-file4.gofile.io/download/zE0DJG/resnet18-binary-classification.pth'
}


def resnet(depth, pretrained=False, progress=True, bin_classif=True, **kwargs):
    """Instantiate a ResNet model for image classification

    Args:
        depth (int): depth of the model
        pretrained (bool, optional): should pretrained parameters be loaded (OpenFire training)
        progress (bool, optional): should a progress bar be displayed while downloading pretrained parameters
        bin_classif (bool, optional): whether the target task is binary classification
        **kwargs: optional arguments of torchvision.models.resnet.ResNet

    Returns:
        model (torch.nn.Module): loaded model
    """

    # Task resolution
    if bin_classif:
        num_classes = 1
    else:
        raise NotImplementedError('architecture not implemented for this task')

    # Model creation
    block = Bottleneck if depth >= 50 else BasicBlock
    model = ResNet(block, resnet_layers[depth], num_classes=num_classes, **kwargs)

    # Parameter loading
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[depth],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model
