import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import pickle
import torchvision
from resnet18 import resnet18
import torch.nn.functional as F

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.RandomHorizontalFlip(),     # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2471, 0.2435, 0.2616)),
])

# Only normalization for test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2471, 0.2435, 0.2616)),
])

# Load CIFAR-10 training and test datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
val_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)


# 定义模型
model = resnet18()
# model.fc = nn.Linear(512, 10)  # 修改最后一层
model = model.to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=3e-4)#optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

# 选择学习率调度器（**选择 StepLR 或 ReduceLROnPlateau 其中之一**）
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每 10 轮学习率衰减
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)  # 依据验证损失调整
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
# 训练函数
def train_model(model, train_loader, val_loader, epochs=30):
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

        # **更新学习率**
        scheduler.step()  # StepLR 的正确用法
        # scheduler.step(val_loss)  # **如果用 ReduceLROnPlateau，使用这个**

        # **打印学习率**
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

# 训练模型
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, epochs=200)

# 保存模型
torch.save(model.state_dict(), "resnet_cifar20.pth")
print("Model saved as resnet_cifar20.pth")



