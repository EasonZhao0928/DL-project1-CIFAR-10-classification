import torch
import torch.nn as nn
import torch.nn.functional as F

# A simplified basic residual block.
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # First convolution: 3x3 kernel, adjustable stride.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolution: 3x3 kernel, stride 1.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output dimensions differ, use a 1x1 conv for downsampling.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return F.relu(out)

# A simplified ResNet18 model.
class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        # Initial convolution: 7x7 kernel, 64 filters, stride 2, padding 3.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Max pooling: 3x3, stride 2, padding 1.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the four layers with default values.
        # Layer1: 2 blocks, 64 channels, stride=1.
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, blocks=2, stride=1)
        # Layer2: 2 blocks, 128 channels, stride=2.
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, blocks=2, stride=2)
        # Layer3: 2 blocks, 256 channels, stride=2.
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, blocks=2, stride=2)
        # Layer4: 2 blocks, 512 channels, stride=2.
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, blocks=2, stride=2)

        # Global average pooling and final fully connected layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block may change dimensions (via stride and channel count).
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Subsequent blocks keep dimensions constant.
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial processing.
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual blocks.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classification.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet(num_classes=10)