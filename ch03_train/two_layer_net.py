import numpy as np
from common.functions import softmax, sigmoid, cross_entropy
from common.gradient import numerical_gradient


# 三层：输入层，隐藏层，输出层
class TwoLayerNet:
    """
    两层神经网络类（包含一个隐藏层）

    网络结构:
        输入层 → 隐藏层(激活函数: Sigmoid) → 输出层(激活函数: Softmax)

    适用于多分类任务，使用交叉熵作为损失函数。
    """

    # 初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化神经网络的参数

        参数:
            input_size (int): 输入层的神经元数量（特征维度）
            hidden_size (int): 隐藏层的神经元数量
            output_size (int): 输出层的神经元数量（类别数量）
            weight_init_std (float): 权重初始化的标准差，默认为0.01
                                    使用较小的值可以避免梯度消失/爆炸

        初始化内容:
            - W1: 输入层到隐藏层的权重矩阵，形状 (input_size, hidden_size)
            - b1: 隐藏层的偏置向量，形状 (hidden_size,)
            - W2: 隐藏层到输出层的权重矩阵，形状 (hidden_size, output_size)
            - b2: 输出层的偏置向量，形状 (output_size,)

        权重初始化策略:
            使用高斯分布随机初始化，乘以小的标准差(0.01)
            这样可以让初始权重接近0但又不全为0，有利于训练
        """
        self.params = {}  # 用字典存储所有参数
        # 第一层权重和偏置
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 第二层权重和偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 定义前向传播（预测）
    def forward(self, X):
        """
        执行前向传播，计算网络的输出

        参数:
            X (numpy.ndarray): 输入数据，形状 (N, input_size)
                              N 是样本数量，input_size 是特征维度

        返回:
            numpy.ndarray: 网络的输出概率分布，形状 (N, output_size)
                          每一行的和为1，表示每个类别的预测概率

        前向传播过程:
            1. 第一层线性变换: a1 = X @ W1 + b1
            2. 第一层激活函数: z1 = sigmoid(a1)
            3. 第二层线性变换: a2 = z1 @ W2 + b2
            4. 第二层激活函数: y = softmax(a2)  ← 输出概率分布

        数学表达:
            隐藏层: z1 = σ(X·W1 + b1)
            输出层: y = softmax(z1·W2 + b2)
            其中 σ 是 sigmoid 函数
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 第一层：输入层 → 隐藏层
        a1 = X @ W1 + b1  # 线性变换（加权求和）
        z1 = sigmoid(a1)  # 激活函数（引入非线性）

        # 第二层：隐藏层 → 输出层
        a2 = z1 @ W2 + b2  # 线性变换
        y = softmax(a2)  # 激活函数（转换为概率分布）

        return y

    # 计算损失
    def loss(self, x, t):
        """
        计算当前模型的损失值（交叉熵损失）

        参数:
            x (numpy.ndarray): 输入数据，形状 (N, input_size)
            t (numpy.ndarray): 真实标签，可以是 one-hot 编码或类别索引
                              如果是 one-hot，形状为 (N, output_size)
                              如果是类别索引，形状为 (N,)

        返回:
            float: 交叉熵损失值，越小表示模型预测越准确

        损失计算流程:
            1. 通过前向传播得到预测结果 y_pred
            2. 计算预测值与真实值的交叉熵

        交叉熵公式:
            L = -Σ t_i · log(y_i)
            其中 t 是真实标签，y 是预测概率
        """
        y_pred = self.forward(x)  # 前向传播得到预测
        loss_value = cross_entropy(y_pred, t)  # 计算交叉熵损失
        return loss_value

    # 计算预测的准确度
    def accuracy(self, x, t):
        """
        计算模型的预测准确率

        参数:
            x (numpy.ndarray): 输入数据，形状 (N, input_size)
            t (numpy.ndarray): 真实标签，形状 (N,)，包含类别索引

        返回:
            float: 准确率（0~1之间），表示预测正确的样本比例

        计算步骤:
            1. 前向传播得到每个类别的概率
            2. 取概率最大的类别作为预测结果
            3. 与真实标签比较，统计正确率

        示例:
            预测概率: [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
            预测类别: [1, 0]  （取每行最大值的索引）
            真实标签: [1, 0]
            准确率:   2/2 = 1.0 （100%）
        """
        y_pred = self.forward(x)
        # 根据最大概率得到预测的分类号
        y = np.argmax(y_pred, axis=1)  # axis=1 表示按行取最大值索引
        # 与正确标签对比，得到准确率
        accuracy = np.sum(y == t) / x.shape[0]  # 正确数 / 总数
        return accuracy

    # 计算梯度
    def numerical_gradient(self, x, t):
        """
        使用数值微分方法计算所有参数的梯度

        参数:
            x (numpy.ndarray): 输入数据，形状 (N, input_size)
            t (numpy.ndarray): 真实标签

        返回:
            dict: 包含所有参数梯度的字典
                  {
                      'W1': W1的梯度，形状同W1,
                      'b1': b1的梯度，形状同b1,
                      'W2': W2的梯度，形状同W2,
                      'b2': b2的梯度，形状同b2
                  }

        梯度含义:
            梯度表示损失函数对每个参数的变化率
            grads['W1'][i,j] 表示: 如果稍微改变 W1[i,j]，损失会如何变化

        计算方法:
            使用中心差分法近似计算偏导数:
            ∂L/∂w ≈ [L(w+h) - L(w-h)] / (2h)
            其中 h 是一个很小的数（如 1e-4）

        注意:
            数值微分计算速度慢，但实现简单，常用于验证反向传播的正确性
            实际训练中应使用反向传播算法（效率高得多）
        """
        # 定义目标函数（固定输入x和标签t，只变化参数w）
        loss_f = lambda w: self.loss(x, t)

        # 对于每个参数，使用数值微分方法计算梯度
        grads = {}
        grads['W1'] = numerical_gradient(loss_f, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_f, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_f, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_f, self.params['b2'])

        return grads
