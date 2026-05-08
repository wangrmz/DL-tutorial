# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0


import numpy as np

# 可以处理向量
'''
特点
支持向量化：输入可以是标量、列表、NumPy 数组等，输出形状与输入相同。

非线性的：输出只有 0 或 1，将输入映射到两个离散值。

不可导：在 0 处不连续，导致无法直接用于基于梯度的优化（所以现代神经网络多用 Sigmoid、ReLU 等可导激活函数）。
'''


def step_function(x):
    # x > 0：对输入x中的每个元素进行大于 0的比较，得到一个布尔类型的数组或标量（True / False）。
    # np.array(..., dtype=int)：将布尔值转换为整数数组，其中True → 1，False → 0。返回值：若x > 0则输出1，否则输出0。
    return np.array(x > 0, dtype=int)


# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU 函数
def relu(x):
    return np.maximum(0, x)


# Softmax
def softmax0(a):
    return np.exp(a) / np.sum(np.exp(a))


# 考虑输入是矩阵的情况
# 将任意实数向量转换为一个有效的概率分布，常用于多分类神经网络的输出层。
# Softmax 将实数向量映射为概率分布，是多分类问题的标准输出。
# 数值稳定性通过“减去最大值”实现，这是工程实现的必备技巧。
# 你的代码正确且高效地处理了单样本与批量样本，并避免了溢出。
# 掌握 Softmax 是理解分类任务、注意力机制、生成模型（如 VAE 输出分布）的基础。

def softmax(x):
    # 如果是二维矩阵
    if x.ndim == 2:
        x = x.T  # 转置为 (K, N)
        x = x - np.max(x, axis=0)   # 减去最大值（标量或向量）
        y = np.exp(x) / np.sum(np.exp(x), axis=0) # 按列归一化 (K, N)
        return y.T # 转回 (N, K)
    # 溢出处理策略
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 恒等函数
def identity(x):
    return x

# 损失函数
# MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差
def cross_entropy(y, t):
    # 将y转为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 将t转换为顺序编码（类别标签）
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]
    return -np.sum( np.log(y[np.arange(n), t] + 1e-10) ) / n



if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print(step_function(x))
    print(sigmoid(x))
    print(np.tanh(x))
    print(relu(x))
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(softmax(X))

