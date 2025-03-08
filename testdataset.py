import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 CIFAR-10 `test_batch` 文件
file_path = "data/cifar-10-batches-py/test_batch"
with open(file_path, "rb") as f:
    data_dict = pickle.load(f, encoding="bytes")  # 读取 pickle 数据

# 2. 提取图片数据和标签
images = data_dict[b"data"]  # 图像数据 (10000, 3072)
labels = data_dict[b"labels"]  # 标签列表

# 3. 重新调整数据形状
num_images = images.shape[0]  # 10000 张图片
image_size = 32
num_channels = 3

# 调整形状为 (10000, 3, 32, 32)
images = images.reshape(num_images, num_channels, image_size, image_size)
# 变换通道顺序为 (10000, 32, 32, 3)，符合 Matplotlib 显示格式
images = images.transpose(0, 2, 3, 1)

# 4. 可视化前 10 张图片
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i in range(10):  # 显示前 10 张图片
    axes[i].imshow(images[i])
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis("off")

plt.show()
