import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import AutoAugmentPolicy
import torch.multiprocessing


#from ResNetTiny import *
from SEPro import *
import os

from model import seresnet18
from model.SEChange import ResNetTiny4

#del model  # 删除模型对象
#torch.cuda.empty_cache()  # 释放 GPU 占用

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.multiprocessing.set_start_method('spawn', force=True)  # 解决 Windows 多进程问题

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomHorizontalFlip(),  # 保留水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 保留颜色抖动
    transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),  # 使用 AutoAugment
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 保留裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

val_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

"""
train_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.2,scale=(0.02, 0.1),value=1.0, inplace=False)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])
"""
# 数据集路径
DATA_DIR = "data/cifar-10"

# 直接使用 torchvision 的 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

# 定义模型
model = ResNetTiny4()
#model.fc = nn.Linear(512, 10)  # 修改最后一层
print(model)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {total_params:,}")

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)


# 训练函数
def train_model(model, train_loader, val_loader, epochs=600):
    best_acc = 0.0  # 记录最佳验证准确率
    best_model_path = "best_resnet_cifar10.pth"

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100 * correct / total

        # 计算验证集损失和准确率
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100 * val_correct / val_total

        # **打印学习率**
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

        scheduler.step()


# 训练模型
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, epochs=600)

# 保存最终模型
torch.save(model.state_dict(), "final_resnet_cifar10.pth")
print("Final model saved as final_resnet_cifar10.pth")




