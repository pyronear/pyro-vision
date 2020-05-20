from torchvision.models.resnet import ResNet, BasicBlock
import torchvision
from torch import nn
import torch
import torch.nn.functional as F


class SSResNet18(ResNet):
    def __init__(self, frame_per_seq):
        super(SSResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        dictres18 = torchvision.models.resnet18(pretrained=True).state_dict()
        self.load_state_dict(dictres18)

        self.frame_per_seq = frame_per_seq
        self.fc = nn.Linear(256, 1)
        self.conv1_1 = nn.Conv2d(512 * self.frame_per_seq, 512, kernel_size=1)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv2_bn1 = nn.BatchNorm2d(512)
        self.conv2_bn2 = nn.BatchNorm2d(256)
        self.sig = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv1_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1_2.weight, mode='fan_out', nonlinearity='relu')

        nn.init.constant_(self.conv2_bn.weight, 1)
        nn.init.constant_(self.conv2_bn.bias, 0)

        nn.init.constant_(self.conv2_bn1.weight, 1)
        nn.init.constant_(self.conv2_bn1.bias, 0)

        nn.init.constant_(self.conv2_bn2.weight, 1)
        nn.init.constant_(self.conv2_bn2.bias, 0)

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

        if x.device.type.startswith('cuda'):
            x = x2.to(x.device)
        else:
            x = x2

        #conv 1x1
        x = self.conv1_1(x)
        x = self.conv2_bn(x)
        x = self.relu(x)
        #conv 3x3
        x = self.conv3_1(x)
        x = self.conv2_bn1(x)
        x = self.relu(x)
        #conv 1x1
        x = self.conv1_2(x)
        x = self.conv2_bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        x2 = torch.cat([x] * self.frame_per_seq)
        for i in range(self.frame_per_seq):
            x2[i::self.frame_per_seq] = x

        return self.sig(x2)
