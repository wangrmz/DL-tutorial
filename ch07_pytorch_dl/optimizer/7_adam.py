import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义函数
def f(X):
    return 0.05 * X[0] ** 2 + X[1] ** 2


# 定义函数实现梯度下降法

def gradient_descent(X, optimizer, num_iters):
    # 拷贝当前X的值，放入列表中
    X_arr = X.detach().numpy().copy()
    for i in range(num_iters):
        # 1.前向转播，得到损失值
        y = f(X)
        # 反向传播获取梯度
        y.backward()
        # 2.更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 更新之后的X, 放入列表中
        # 堆叠
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return  X_arr

# 主流程
# 1.参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 定义超参数
lr = 0.1
num_iters = 500

'''
分解说明：
方法                    作用.                 为什么要这样做？
X.clone()              深拷贝张量             创建一个新的张量，与原张量共享数据但独立存储
.detach()               分离计算图            从原来的计算图中分离出来，不保留之前的梯度历史
.requires_grad_(True)  启用梯度追踪           让新张量能够记录梯度，用于反向传播
'''

# 3优化器对比
# 3.2 SGD
# 为SGD优化器创建独立的副本
# clone() 复制张量，detach() 分离计算图，requires_grad_(True) 启用梯度追踪
X_clone = X.clone().detach().requires_grad_(True)

optimizer = torch.optim.SGD([X_clone], lr=lr)

# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)

# 画出点轨迹
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'b')


# 3.2  Adam
'''
7.5.5Adam
Adam（Adaptive Moment Estimation，自适应矩估计）融合了Momentum和AdaGrad的方法。

可以通过torch.optim. Adam ()并设置betas权重参数元组，其中包含两个权重参数，来使用Adam

'''
# 重新创建一个新的克隆张量，避免与SGD共享内存
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([X_clone], lr=lr,betas=(0.9,0.999))

# 梯度下降
X_arr2 = gradient_descent(X_clone, optimizer, num_iters)

# 画出点轨迹
plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'r')

# 画出等高线

# 绘制等高线图
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid ** 2 + x2_grid ** 2

plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors="gray")
plt.legend(["SGD", "Adam"])
plt.show()



















