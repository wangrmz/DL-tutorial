import numpy as np
import pandas as pd
import joblib  # 加载之前训练好的神经网络参数（权重和偏置）
from common.functions import sigmoid, softmax  # 激活函数
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# 读取数据
def get_data():
    # 1.读取数据集
    data = pd.read_csv('../data/train.csv')
    # 2.划分数据集
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3.特征工程：归一化处理
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 模型是训练好的，所以目前只需要测试数据
    return X_test, y_test


# 初始化神经网络
# joblib.load 读取之前保存的神经网络参数字典。该字典通常包含：
# W1, W2, W3：各层的权重矩阵。
# b1, b2, b3：各层的偏置向量。
# 模型结构是一个三层神经网络（输入层 → 隐藏层1 → 隐藏层2 → 输出层，共3个权重层）。输出层使用 softmax。

def init_network():
    # 直接从文件中读取
    network = joblib.load("../data/nn_sample")
    return network


# 前向传播
'''
输入 x 的形状为 (样本数, 输入特征数)。

第一层：线性变换 a1 = x·W1 + b1，再经过 sigmoid 激活得到 z1。

第二层：同样线性变换 + sigmoid，得到 z2。

第三层（输出层）：线性变换 a3 = z2·W3 + b3，再经过 softmax 激活，将输出转换成每个类别的概率（所有概率之和为1）。

返回概率矩阵 y，形状为 (样本数, 类别数)。
'''


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算传播
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)  # 激活函数 sigmoid 用于隐藏层，softmax 用于输出层（多分类）。
    return y


# 主流程
# 1.获取数据
x, y = get_data()
print(x.shape, y.shape)

# 2.创建模型（加载参数）
network = init_network()
# print(network['W1'].shape)
# print(network['W2'].shape)
# print(network['W3'].shape)
# print(network['b1'].shape)
# print(network['b2'].shape)
# print(network['b3'].shape)

# 3.前向传播
'''
(8400, 10) —— 表示 8400 个样本，每个样本对 10 个类别的预测概率（说明是个10分类任务，如手写数字识别 MNIST）。
'''
# y_proda = forward(network, x)
# print(y_proda.shape) # (8400, 10)
#
# # 4.将分类概率转换为分类标签
# # argmax 按行（axis=1）取最大概率的索引，得到预测的类别标签（0~9）。
# y_pred = np.argmax(y_proda, axis=1)
#
# # 5.计算分类准确率
# accuracy_cnt = np.sum(y_pred == y) # 预测准确的数量
# n = x.shape[0]
# print('准确率:', accuracy_cnt / n) # 准确率: 0.9353571428571429


# 定义变量
batch_size = 100
accuracy_cnt = 0
# 样本数量
n = x.shape[0]

# 3.循环迭代：分批次坐测试，前向传播，并累计预测正确的数量
for i in range(0, n, batch_size):
    # 3.1 取出当前批次的数据
    x_batch = x[i:i + batch_size]
    # 3.2  前向传播
    y_batch = forward(network, x_batch)
    y_true = y[i:i + batch_size]
    # 3.3 将输出分类概率转换为分类标签
    y_pred = np.argmax(y_batch, axis=1)
    # 3.4 累计预测正确的数量
    accuracy_cnt += np.sum(y_pred == y_true)

print('准确率:', accuracy_cnt / n) # 准确率: 0.9353571428571429






















