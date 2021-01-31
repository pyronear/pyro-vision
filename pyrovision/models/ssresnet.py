# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from torchvision.models.resnet import ResNet, BasicBlock
from torch import nn
import torch


__all__ = ["SSResNet"]


class SSResNet(ResNet):
    """This model is designed to be trained using the SubSamplerDataSet. It can be built over any resnet.
    The SubSamplerDataSet will send within the same batch K consecutive frames belonging to the same
    sequence. The SSresnet model will process these K frames independently in the first 4 layers of
    the resnet then combine them in a 5th layer.
    Args:

    To build a Resnet we need two arguments, are we using a BasicBlock or a Bottleneck and
    the corresponding layers. This is how to build the ResNets:
    resnet18: BasicBlock, [2, 2, 2, 2]
    resnet34: BasicBlock, [3, 4, 6, 3]
    resnet50: Bottleneck, [3, 4, 6, 3]
    resnet101: Bottleneck, [3, 4, 23, 3]
    resnet152: Bottleneck, [3, 8, 36, 3]
    Please refere to torchvision documentation for more details:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L232
        block (string): BasicBlock or Bottleneck
        layers (list): layers argument to build BasicBlock / Bottleneck
        frame_per_seq (int): Number of frame per sequence
    Then we need shapes of the layer5
        shapeAfterConv1_1 (int): Output shape of the first conv1x1
        outputShape (int): Output shape of the second conv1x1
    """
    def __init__(self, block, layers, frame_per_seq=2, shapeAfterConv1_1=512, outputShape=256):

        super(SSResNet, self).__init__(block, layers)

        self.frame_per_seq = frame_per_seq

        self.layer5 = self._make_layer5(intputShape=512 * block.expansion, shapeAfterConv1_1=shapeAfterConv1_1,
                                        outputShape=outputShape)

        for m in self.layer5.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fc = nn.Linear(256, 1)

    def _make_layer5(self, intputShape, shapeAfterConv1_1, outputShape):

        layer5 = nn.Sequential(nn.Conv2d(intputShape * self.frame_per_seq, shapeAfterConv1_1, kernel_size=1),
                               nn.BatchNorm2d(shapeAfterConv1_1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(shapeAfterConv1_1, shapeAfterConv1_1, kernel_size=3),
                               nn.BatchNorm2d(shapeAfterConv1_1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(shapeAfterConv1_1, outputShape, kernel_size=1),
                               nn.BatchNorm2d(outputShape),
                               nn.ReLU(inplace=True),
                               )

        return layer5

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x2 = torch.zeros((x.shape[0] // self.frame_per_seq, x.shape[1] * self.frame_per_seq, x.shape[2], x.shape[3]))
        for i in range(x.shape[0]):
            s = i % self.frame_per_seq
            x2[i // self.frame_per_seq, s * x.shape[1]:(s + 1) * x.shape[1], :, :] = x[i, :, :, :]

        x = x2.to(x.device)

        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        x2 = torch.cat([x] * self.frame_per_seq)
        for i in range(self.frame_per_seq):
            x2[i::self.frame_per_seq] = x

        return x2


def ssresnet18(frame_per_seq=2, **kwargs):
    r"""SubSamplerResNet18 from ResNet-18 model

    Args:
        frame_per_seq (int, optional): Number of frame per sequence
    """
    return SSResNet(BasicBlock, [2, 2, 2, 2], frame_per_seq=frame_per_seq)
