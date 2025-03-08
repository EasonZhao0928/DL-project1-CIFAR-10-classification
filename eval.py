import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torch.nn as nn
from resnet18 import resnet18

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载训练好的 ResNet 模型
# model = models.resnet18(pretrained=False)
model = resnet18()
# model.fc = nn.Linear(512, 10)  # 适配 CIFAR-10 10 类
model.load_state_dict(torch.load("./resnet_cifar20.pth"))  # 请确保模型已保存
model = model.to(device)
model.eval()

# 加载自定义测试集
test_data_path = "./cifar_test_nolabel.pkl"
with open(test_data_path, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

test_data = test_dict[b'data']  # (10000, 32, 32, 3)

test_data = torch.tensor(test_data, dtype=torch.float32)  # Shape: [N, H, W, C]
test_data = test_data.permute(0, 3, 1, 2)  # Now shape: [N, C, H, W]

# Normalize to [0, 1]
test_data = test_data / 255.0

# CIFAR-10 mean and std values (for each channel)
mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
std  = torch.tensor([0.2471, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)

# Apply normalization: (x - mean) / std
test_data = (test_data - mean) / std

# test_data now is normalized according to CIFAR-10 statistics.
# print(test_data.shape)  # Should be [N, 3, H, W]


test_dataset = TensorDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 预测
test_ids = list(range(test_data.shape[0]))
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        predictions.extend(predicted_labels.cpu().numpy())

# 生成 CSV 提交文件
submission = pd.DataFrame({"ID": test_ids, "Labels": predictions})
submission.to_csv("submission2.csv", index=False)
print("Test predictions saved to submission2.csv")