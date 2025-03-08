import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import transforms
from Resnet import *

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预处理（确保和训练时一致）
test_transform = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载训练好的 ResNet 模型
model = ResNet18()
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load("./resnet_cifar10.pth"))  # **先加载权重**
model = model.to(device)
model.eval()  # **确保关闭 Dropout**

# 加载测试数据
test_data_path = "./cifar_test_nolabel.pkl"
with open(test_data_path, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

test_data = test_dict[b'data']  # (10000, 32, 32, 3)
test_data = np.transpose(test_data, (0, 3, 1, 2))  # 变为 (10000, 3, 32, 32)

# 转换为张量 & 预处理
test_data = torch.tensor(test_data, dtype=torch.float32) / 255.0
test_data = torch.stack([test_transform(img) for img in test_data])  # 应用标准化

# DataLoader
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# 预测
test_ids = list(range(test_data.shape[0]))
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
        predictions.extend(predicted_labels.cpu().numpy())

# 生成 CSV 提交文件
submission = pd.DataFrame({"ID": test_ids, "Labels": predictions})
submission.to_csv("submission5.csv", index=False)
print("Test predictions saved to submission5.csv")
