import torch
import matplotlib.pyplot as plt
from torch import nn, optim  # 模型、损失函数和优化器
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器

# 1.构建数据集，创建数据加载器
X = torch.randn(100, 1)  # 输入

# 预设真实系数
w = torch.tensor([2.5])  # 权重
b = torch.tensor([5.2])  # 偏置
# 定义随机噪声
noise = torch.randn(100, 1) * 0.1  # 噪声
# 定义拟合的目标值y
y = w * X + b + noise
# 构造数据集对象
dataset = TensorDataset(X, y)
# 构造数据加载器对象，batch_size为每次训练的样本数，shuffle为是否打乱数据
# 本质上就是一个小批次数据
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2.构造模型
# 一元线性回归，一个输入神经元，一个输出神经元
model = nn.Linear(1, 1)

# 3.定义损失函数和优化器
loss = nn.MSELoss()
# 优化器，模型的参数，学习率超参数
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 4.模型训练
epoch_num = 1000
loss_list = []
# 训练epoch_num轮次
for epoch in range(epoch_num):
    # 本轮总损失
    total_loss = 0
    # 本轮迭代次数
    iter_num = 0
    # 一个轮次，遍历DataLoader对象，得到一个批次的数据
    for x_train, y_train in dataloader:
      # 4.1 前向传播（预测）
      y_pred = model(x_train)
      # 4.2 计算损失
      loss_value = loss(y_pred, y_train)
      total_loss += loss_value.item() # 需要转成标量
      iter_num += 1
      # 4.3 反向传播（计算梯度）
      loss_value.backward()
      # 4.4更新参数
      optimizer.step() # 使用优化器，更新参数
      # 4.5 清零梯度
      optimizer.zero_grad()
    # 计算本轮次的平均损失
    loss_list.append(total_loss / iter_num)


# 打印参数
print(model.weight)
print(model.bias)
'''
Parameter containing:
tensor([[2.5085]], requires_grad=True)
Parameter containing:
tensor([5.1964], requires_grad=True)
'''

# 画图
fig ,ax = plt.subplots(1,2 , figsize=(12,4))

# 1. 训练损失随轮次epoch的变化
ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].set_title('Training Loss')
# 2. 预测值和真实值
ax[1].scatter(X, y, label='true')

y_pred = model.weight.item() * X + model.bias.item()
ax[1].plot(X, y_pred, label='pred')
plt.show()

















