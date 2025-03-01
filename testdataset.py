import pickle

# 文件路径
test_data_path = "./cifar_test_nolabel.pkl"

# 读取 pkl 文件
with open(test_data_path, 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

# 打印 pkl 文件的键
print("Keys in the pickle file:", test_dict.keys())

# 预览数据内容（仅显示前5个样本）
if b'data' in test_dict:
    print("Shape of data:", test_dict[b'data'].shape)
    print("First 5 samples:", test_dict[b'data'][:5])

# 检查是否有 ID 信息
if b'ID' in test_dict:
    print("First 5 IDs:", test_dict[b'ID'][:5])