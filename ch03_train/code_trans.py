import numpy as np

# ========== 顺序编码 → One-hot编码 ==========
t_sequential = np.array([0, 2, 1, 0])
num_classes = 3

# 方法: 使用np.eye创建单位矩阵,然后索引
t_onehot = np.eye(num_classes)[t_sequential]
print(t_onehot)
# 输出:
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]

print("One-hot编码 → 顺序编码")
# ========== One-hot编码 → 顺序编码 ==========
t_onehot = np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [1, 0, 0]])

# 方法: 取每行最大值的索引
t_sequential = np.argmax(t_onehot, axis=1)
print(t_sequential)
# 输出: [0, 2, 1, 0]
