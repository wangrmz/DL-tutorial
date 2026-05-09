import numpy as np
from common.functions import softmax, cross_entropy
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def forward(self, x):
        return softmax(x @ self.W)

    def loss(self, x, t):
        return cross_entropy(self.forward(x), t)


# 准备数据
net = SimpleNet()
x = np.array([[0.6, 0.9]])
t = np.array([0, 0, 1])
lr = 0.1

print("开始训练...\n")

# 记录历史
losses = []
grads = []

for i in range(100):
    # 步骤1: 计算当前损失
    loss = net.loss(x, t)
    losses.append(loss)

    # 步骤2: 计算当前梯度
    f = lambda w: net.loss(x, t)
    gradw = numerical_gradient(f, net.W)
    grads.append(np.mean(np.abs(gradw)))

    # 步骤3: 更新权重
    net.W -= lr * gradw

    # 打印关键信息
    if i % 20 == 0:
        y_pred = net.forward(x)
        print(f"迭代 {i:3d}:")
        print(f"  损失: {loss:.6f}")
        print(f"  平均梯度: {grads[-1]:.6f}")
        print(f"  预测: {y_pred.flatten()}")
        print(f"  真实: {t}")
        print()

print("\n训练完成!")
print(f"最终损失: {losses[-1]:.6f}")
print(f"初始损失: {losses[0]:.6f}")
print(f"损失降低了: {(1 - losses[-1] / losses[0]) * 100:.2f}%")
