import numpy as np
from common.functions import softmax, cross_entropy # 将输出转换为概率分布,计算预测值与真实值的差距
from common.gradient import numerical_gradient #  用数值微分方法计算梯度


# 定义一个简单神经网络类
# 输入层是2个神经元，输出层是3个神经元
# 输入层: 2个神经元 (特征维度为2)
# 输出层: 3个神经元 (3个类别)
# 激活函数: Softmax
# 损失函数: 交叉熵
class SimpleNet:
    """
    定义一个简单的神经网络类，用于预测标签。
    """

    def __init__(self):
        '''
        创建权重矩阵 W,形状为 (2, 3)
        randn 生成标准正态分布的随机数
        为什么是 2×3?
        输入有2个特征 → 2行
        输出有3个类别 → 3列
        '''
        self.W = np.random.randn(2, 3)

    def forward(self, x):
        a = x @ self.W
        y = softmax(a)
        return y

    # 计算损失值
    def loss(self, x, t):
        y_predict = self.forward(x)
        # 计算交叉熵损失
        loss = cross_entropy(y_predict, t)
        return loss


# 主流程
if __name__ == '__main__':
    # 1.定义数据
    x = np.array([[0.6, 0.9]])  # 输入样本: 1个样本,2个特征
    t = np.array([0, 0, 1])  # 真实标签: one-hot编码,表示第3类
    # 2.定义模型
    net = SimpleNet()  # 实例化网络,随机初始化权重 W
    # 3.计算梯度
    # 定义一个函数，用于计算损失值
    # f(w) = net.loss(x, t)  # 固定 x 和 t,只变化 w
    f = lambda w: net.loss(x, t)  # 定义损失函数
    '''
    如果 gradw[i,j] > 0:  减小 w[i,j] 可以降低损失
    如果 gradw[i,j] < 0:  增大 w[i,j] 可以降低损失
    更新规则 (梯度下降):
    W_new = W_old - learning_rate × gradw
    '''
    # numerical_gradient 内部会对 W 的每个元素 w_ij 计算:
    # ∂L /∂w_ij ≈ [L(W + h·e_ij) - L(W - h·e_ij)] / (2h)
    # 其中 e_ij 是只有第(i,j)位置为1,其余为0的矩阵

    # 梯度结果: gradw
    # 的形状也是(2, 3), 表示每个权重对损失的影响程度
    gradw = numerical_gradient(f, net.W)  # 计算权重 W 的梯度
    print(gradw)

'''
梯度 = 损失函数对权重的变化率
通俗说:
- 梯度告诉你:"如果稍微改变某个权重,损失会怎么变"
- 梯度大 → 损失对这个权重很敏感
- 梯度小 → 损失对这个权重不敏感
- 梯度为0 → 损失达到极值点(最小或最大)

'''
# 完整的训练还需要:梯度更新
# learning_rate = 0.01
# net.W = net.W - learning_rate * gradw  # ← 这里用负梯度!
#
# # 重复多次迭代
# for i in range(1000):
#     gradw = numerical_gradient(f, net.W)
#     net.W -= learning_rate * gradw  # 梯度下降

'''
输入 x=[0.6, 0.9]
         ↓
    ┌──────────┐
    │  W (2×3) │  ← 随机初始化的权重
    └──────────┘
         ↓
    x @ W (线性变换)
         ↓
    a (3个输出值)
         ↓
   softmax (激活函数)
         ↓
    y_pred (概率分布)
         ↓
  cross_entropy (与真实标签比较)
         ↓
    损失值 L
         ↓
numerical_gradient (计算梯度)
         ↓
   gradw (2×3梯度矩阵)

'''

'''
为什么要计算梯度?梯度用来指导参数更新方向
梯度告诉我们:如何调整权重才能让损失减小

如果 gradw[i,j] > 0:  减小 w[i,j] 可以降低损失
如果 gradw[i,j] < 0:  增大 w[i,j] 可以降低损失

更新规则 (梯度下降):
W_new = W_old - learning_rate × gradw

'''

'''
梯度指向增大方向

> "在寻找损失函数的最小值的时候，不是一般选负梯度吗?"

✅ **对!** 所以更新公式是 `W = W - lr × grad` (减号就是取负梯度)

> "损失函数小不是表明参数选择比较合理吗?"

✅ **对!** 损失小 → 预测接近真实 → 参数好


梯度定义: ∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]

性质:
- 梯度 ∇L 指向 L 增大最快的方向
- 负梯度 -∇L 指向 L 减小最快的方向

目标: 最小化损失函数 L
方法: 沿着负梯度方向更新参数

更新公式:
W_new = W_old - learning_rate × ∇L
         ↑       ↑              ↑
       新权重   旧权重    学习率×梯度(负号已体现)

'''
