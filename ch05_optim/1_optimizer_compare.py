from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


# 定义目标函数：f(x,y) = 1/20 x ^ 2  + y ^ 2
def f(x, y):
    return x ** 2 / 20 + y ** 2


# 定义梯度计算方法，得到一个长度为2 的向量
def f_gradient(x, y):
    return x / 10, y * 2


# 定义初始点的位置
init_pos = (-7.0, 2)

# 定义当前的参数和梯度
params = {}
grads = {}

# 定义优化器，指定学习率
# 定义有序字典
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.9)
optimizers['Momentum'] = Momentum(lr=0.08, momentum=0.9)
optimizers['AdaGrad'] = AdaGrad(lr=1)
optimizers['Adam'] = Adam(lr=0.5)

# 定义绘图
idx = 1  # 子图序号

for key in optimizers:
    optimizer = optimizers[key]
    # 记录参数点更新的历史
    x_history = []
    y_history = []
    # 参数初始化
    params['x'], params['y'] = init_pos[0], init_pos[1]
    # 指定迭代 30 次
    for i in range(30):
        # 记录参数点
        x_history.append(params['x'])
        y_history.append(params['y'])

        # 1.计算梯度
        grads['x'],grads['y'] = f_gradient(params['x'], params['y'])
        # 2.更新参数
        optimizer.update(params, grads)



    # 画图
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y) # 等高线
    Z = f(X, Y)
    Z[Z>7] = 0 # 布尔索引，大于7的部分不画

    # 定义子图
    plt.subplot(2, 2, idx)
    idx += 1
    # 绘制等高线
    plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35))
    # 单独画出最小值
    plt.plot(0,0,'+')
    # 画出点轨迹曲线
    plt.plot(x_history, y_history, 'o-', color='red',markersize =2,label = key)


    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend(loc='best')

plt.show()

























