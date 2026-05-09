import numpy as np
import matplotlib.pyplot as plt
from common.load_data import get_data
from two_layer_net import TwoLayerNet

# 1.加载数据
X_test, X_train, y_test, y_train = get_data()

# 2.创建模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3.设置超参数
learning_rate = 0.1
batch_size = 100
num_epochs = 10

train_size = X_train.shape[0]
iterations_per_epoch = np.ceil(train_size / batch_size)
iters_num = int(num_epochs * iterations_per_epoch)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 4.循环迭代，用梯度下降法训练
for i in range(iters_num):
    # 4.1 随机选取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = y_train[batch_mask]
    # 4.2 计算梯度（非常耗时）
    # grad = network.numerical_gradient(x_batch, t_batch)
    # 使用反向传播计算梯度
    grad = network.gradient(x_batch, t_batch)
    # print('梯度grad====：', i)
    # 4.3更新梯度
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 4.4 计算并保存当前的训练损失
    train_loss = network.loss(x_batch, t_batch)
    train_loss_list.append(train_loss)
    # 4.5 计算并保存当前的训练准确率（按照轮次来保存，每完成一个epoch保存）
    if i % iterations_per_epoch == 0:
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('epoch:', i // iterations_per_epoch + 1,
              'train acc:', train_acc, 'test acc:', test_acc)

# 5.画图
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.legend(loc='best')
plt.show()

'''
Epoch ( epoch/轮次)
定义: 整个训练集被完整遍历一次

例如:
- 训练集有 60,000 张图片
- 1个epoch = 模型看过所有60,000张图片一次
- 通常训练需要多个epoch(如10个epoch = 看10遍)

'''

'''
Batch Size (批次大小)
定义: 每次更新参数时使用的样本数量

例如:
- batch_size = 100
- 每次从训练集中取100个样本计算梯度并更新权重
- 而不是一次性用全部60,000个样本
'''

'''
Iteration (迭代次数)
定义: 完成一个epoch需要的批次数

计算公式:
iterations_per_epoch = 总样本数 / batch_size

例如:
- 训练集60,000个样本, batch_size=100
- iterations_per_epoch = 60,000 / 100 = 600次迭代
- 即一个epoch需要600次参数更新

'''

'''

含义:
每个epoch需要 480次迭代(每次处理100个样本)
总共10个epoch,所以总共需要 4800次迭代

train_size = X_train.shape[0]              # 假设48,000个训练样本
iterations_per_epoch = np.ceil(train_size / batch_size)  # ceil(48000/100) = 480
iters_num = num_epochs * iterations_per_epoch            # 10 * 480 = 4800

'''

'''
损失函数曲面:

BGD (全量):          SGD (单样本):        Mini-batch (批量):
    ↓                    ↓                     ↓
   ━━━                  ~~~                   ≈≈≈
  平滑下降             剧烈震荡               小幅波动
  (稳定但慢)           (快但不稳)             (平衡)

'''

'''
原因1: 内存效率
原因2: 计算效率
# GPU并行计算优势
- 单个样本: 无法充分利用GPU并行能力
- 小批量: 可以矩阵运算,充分利用GPU
- 全量: 可能超出显存

原因3: 泛化能力
# 小批量的噪声有助于跳出局部最优解
# 就像下山时有点"抖动",反而能找到更好的路径


'''
