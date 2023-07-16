import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip_connection = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __int__(self):
        super().__init__()
        self.Conv2D = nn.Conv2d(3, 64, 7, 2)
        self.BatchNorm = nn.BatchNorm1d()
        self.Relu = nn.ReLU()
        self.MaxPool = nn.MaxPool1d(3, 2)
        self.ResBlock = self._make_resblock(64, 64, 1)
        self.ResBlock = self._make_resblock(64, 128, 2)
        self.ResBlock = self._make_resblock(128, 256, 2)
        self.ResBlock = self._make_resblock(256, 512, 2)
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(512, 2)
        self.Sigmoid = nn.Sigmoid()

    def _make_resblock(self, in_channels, out_channels, stride):
        layers = [ResBlock(in_channels, out_channels, stride), ResBlock(out_channels, out_channels, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)

        out = self.global_avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out
