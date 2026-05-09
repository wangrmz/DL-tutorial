
import numpy as np
from collections import OrderedDict  # 有序字典，保证层的插入顺序
from common.functions import softmax, sigmoid, cross_entropy
from common.gradient import numerical_gradient
from common.layer import * # 引入神经网络的层



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
        # 定义层结构,有序字典，保证插入顺序
        # 关键点:
        # ✅ 前向传播必须按顺序: 输入 → Affine1 → ReLU1 → Affine2 → 输出
        # ✅ 反向传播必须逆序: 输出梯度 → Affine2 → ReLU1 → Affine1 → 输入梯度
        # ✅ OrderedDict
        # 保证了这个顺序的正确性
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 单独定义最后一层：SoftmaxWithLoss
        self.lastLayer = SoftmaxWithLoss()


    # 定义前向传播（预测）
    def forward(self, X):
        # 对于神经网络层中方的每一层，依次调用forward方法
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    # 计算损失
    def loss(self, x, t):
        y_pred = self.forward(x)  # 前向传播得到预测
        loss_value = self.lastLayer.forward(y_pred, t)  # 计算损失
        return loss_value

    # 计算预测的准确度
    def accuracy(self, x, t):
        y_pred = self.forward(x) # 预测分类数值
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
    # 计算梯度，反向传播
    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)
        # 反向传播
        dy = 1
        dy = self.lastLayer.backward(dy)
        # 将神经网络中的所有层翻转处理
        for layer in reversed(self.layers.values()):
            dy = layer.backward(dy)
        # 提取各层参数的梯度
        grads = {
                'W1': self.layers['Affine1'].dW,
                'b1': self.layers['Affine1'].db,
                'W2': self.layers['Affine2'].dW,
                'b2': self.layers['Affine2'].db
        }
        return grads

