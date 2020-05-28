from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torchvision
from torch import nn
import torch
import torch.nn.functional as F


class SSResNet(ResNet):
    """This model is designed to be trained using the SubSamplerDataSet. It can be built over any resnet.
       The SubSamplerDataSet will send within the same batch K consecutive frames belonging to the same
       sequence. The SSresnet model will process these K frames independently in the first 4 layers of
       the resnet then combine them in a 5th layer.
    """
    def __init__(self, frame_per_seq, block='BB', layers=[2, 2, 2, 2], pretrainedWeights=False, intput1L5=512,
                 output1L5=512, output2L5=256):
        """ Init SSResNet

            Attributes
            ----------
            frame_per_seq: int
                Number of frame per sequence

            To build a Resnet we need two arguments, are we using a BasicBlock or a Bottleneck and
            the corresponding layers. This is how to build the ResNets:
            resnet18: BasicBlock, [2, 2, 2, 2]
            resnet34: BasicBlock, [3, 4, 6, 3]
            resnet50: Bottleneck, [3, 4, 6, 3]
            resnet101: Bottleneck, [3, 4, 23, 3]
            resnet152: Bottleneck, [3, 8, 36, 3]

            Please refere to torchvision documentation for more details:
            https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L232

            block: string
                block: BasicBlock or Bottleneck

            layers: list
                layers argument to build BasicBlock / Bottleneck

            pretrainedWeights: bool
                pretrained Weights for the ResNet

            Then we need shapes of the layer5

            intput1L5: int
                Input of the layer5, must be equal to the output of layer4

            output1L5: int
                Output shape of the first conv1x1

            output2L5: int
                Output shape of the second conv1x1

            """
        if block == 'BB':
            super(SSResNet, self).__init__(BasicBlock, layers)
        else:
            super(SSResNet, self).__init__(Bottleneck, layers)

        if pretrainedWeights:
            self.load_state_dict(pretrainedWeights)

        self.frame_per_seq = frame_per_seq

        self.layer5 = self._make_layer5(intput1=intput1L5, output1=output1L5, output2=output2L5)

        for m in self.layer5.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fc = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def _make_layer5(self, intput1, output1, output2):

        layer5 = nn.Sequential(nn.Conv2d(intput1 * self.frame_per_seq, output1, kernel_size=1),
                               nn.BatchNorm2d(output1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(output1, output1, kernel_size=3),
                               nn.BatchNorm2d(output1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(output1, output2, kernel_size=1),
                               nn.BatchNorm2d(output2),
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

        return self.sig(x2)
