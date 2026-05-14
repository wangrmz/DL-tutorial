import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体：避免绘图中中文乱码，Heiti TC是macOS下的黑体；关闭unicode减号显示问题。
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 从(-7, 2)出发
# 1.参数X初始化
X = torch.tensor([-7, 2], dtype=torch.float32, requires_grad=True)
w = torch.tensor([[0.05], [1.0]], requires_grad=True)
# 定义超参数
lr = 0.9  # 初始学习率
n_iters = 1000  # 迭代次数

# 定义优化器 SGD
optimizer = torch.optim.SGD([X], lr=lr)
scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # 学习率衰减
X_arr = X.detach().numpy().copy()  # 拷贝，用于记录优化过程
lr_list = []  # 记录学习率变化
for epoch in range(n_iters):
    # 1.前向转播，得到损失值
    y = X ** 2 @ w
    # 2 反向传播获取梯度
    y.backward()
    # 3.更新参数
    optimizer.step()
    optimizer.zero_grad()  # 清空梯度
    X_arr = np.vstack([X_arr, X.detach().numpy()])  # 记录优化过程
    lr_list.append(optimizer.param_groups[0]["lr"])  # 记录学习率变化
    # 学习率衰减
    scheduler_lr.step()

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = w.detach().numpy()[0, 0] * x1_grid ** 2 + w.detach().numpy()[1, 0] * x2_grid ** 2

# 画等高线和X点的轨迹
ax[0].contour(x1_grid, x2_grid, y_grid, levels=30, colors="gray")
# 第一列，第二列
ax[0].plot(X_arr[:, 0], X_arr[:, 1], "r")
ax[0].set_title("梯度下降过程")

# 画学习率衰减曲线
ax[1].plot(lr_list, "k")
ax[1].set_title("学习率衰减")
plt.show()
