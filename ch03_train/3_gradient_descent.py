import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_gradient


# 定义梯度下降法的的函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    # 定义列表保存x的变化
    x_history = []
    # 循环迭代
    for i in range(step_num):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 更新参数
        x -= lr * grad
    return x, np.array(x_history)


# 定义目标函数：f(x1,x2) = x1^2 + x2^2
def f(x):
    return x[0] ** 2 + x[1] ** 2


# 主流程
if __name__ == '__main__':
    # 1.定义初始值
    init_x = np.array([-3.0, 4.0])
    # 2.定义超参数
    lr = 0.01
    num_iter = 200  # 迭代次数
    # 3.使用梯度下降法，计算最小值点
    x, x_history = gradient_descent(f, init_x, lr, num_iter)
    print(f"最小值点: {x}")

    # 画图

    # x_history = [
    #     [-3.0, 4.0],  # 第0次迭代的位置
    #     [-2.4, 3.2],  # 第1次迭代的位置
    #     [-1.92, 2.56],  # 第2次迭代的位置
    #     ...
    #     [-0.05, 0.07]  # 第19次迭代的位置
    # ]
    # x_history的形状: (20, 2)，20次迭代每个x有2个维度
    plt.plot([-5, 5], [0, 0], color='blue', linestyle='--')  # 水平线 (x轴)
    plt.plot([0, 0], [-5, 5], color='blue', linestyle='--')  # 垂直线 (y轴)

    plt.plot([-5, 5], [0, 0], 'b--')  # 蓝色虚线
    plt.plot([0, 0], [-5, 5], 'b--')  # 蓝色虚线

    plt.axhline(y=0, color='blue', linestyle='--', alpha=0.5)  # 水平线
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.5)  # 垂直线

    # x_history[:, 0] 是什么?所有迭代点的 x1 坐标
    # x_history[:, 1] 是什么?所有迭代点的x2坐标
    plt.scatter(x_history[:, 0], x_history[:, 1], marker='x')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.show()
