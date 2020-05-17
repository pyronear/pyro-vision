from torchvision.models.resnet import ResNet, BasicBlock
from torch import nn
import torch
import torch.nn.functional as F


class SSResNet18(ResNet):
    def __init__(self, frame_per_seq):
        super(SSResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.frame_per_seq = frame_per_seq
        self.fc = nn.Linear(int(512 * self.frame_per_seq), 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        bs = x.shape[0]
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #merge data from same seq
        x = x.view(int(bs / self.frame_per_seq), int(512 * self.frame_per_seq))

        x = F.relu(self.fc(x))
        x = self.fc2(x)
        # reshape to expexted output
        x2 = torch.cat([x] * self.frame_per_seq)
        for i in range(self.frame_per_seq):
            x2[i::self.frame_per_seq] = x

        return x2
