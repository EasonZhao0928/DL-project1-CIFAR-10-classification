
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.global_avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out

class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)  # 加入 SE 机制

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 经过 SE 模块
        return F.relu(out + x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetTiny(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetTiny, self).__init__()
        self.in_planes = 48  # 初始通道数

        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 调整全连接层的输入通道数，减少模型参数
        self.linear = nn.Linear(256 * block.expansion, num_classes)  # 256是最后一层通道数

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNetTiny1():
    # 修改网络结构来减少参数量，减少深度和每层的卷积通道
    return ResNetTiny(SEBasicBlock, [1, 1, 2, 2])  # 减少层数，增加性能表现
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetTiny1().to(device)

summary(model, (3, 32, 32))  # 假设输入为 3x32x32 图像


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 测试模型
model = ResNetTiny()
print(f"参数量: {count_params(model) / 1e6}M")


def test():
    net = ResNetTiny()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

if __name__ == "__main__":
    test()
"""
