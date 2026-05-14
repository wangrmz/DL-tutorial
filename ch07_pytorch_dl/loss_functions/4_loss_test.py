import torch
from torch import nn, optim


# 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义一个全连接层
        self.linear = nn.Linear(5, 3)
        # 权重初始化
        self.linear.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.6, 0.7, 0.8],
                [1.1, 1.2, 1.3],
                [1.6, 1.7, 1.8],
                [1.6, 1.7, 1.8]
            ]
        ).T

        # 偏置初始化
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3])


    def forward(self, x):
        x = self.linear(x)
        return x

# 主流程
# 1 定义数据 输入数据 2 * 5
X = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]], dtype=torch.float)

#  目标值2 * 3
target = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float)

#  2.创建模型
model = Model()

# 3.前向传播，预测输出
output = model(X)

# 4.定义损失函数，计算损失
loss = nn.MSELoss()
loss_value = loss(output, target)

# 5.计算梯度
loss_value.backward()

# 6.定义一个优化器【针对模型参数进行优化】
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 7.更新参数
optimizer.step()
optimizer.zero_grad() # 清零梯度

# print(model.linear.weight)

# 打印模型参数
# for name, param in model.named_parameters():
#     print(f"{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
#

for param in model.state_dict():
    print(param) # 参数名字
    print(model.state_dict()[param]) # 参数值


'''

linear.weight
tensor([[-0.8457, -0.5563, -0.2670,  0.0223, -0.1883],
        [-0.8330, -0.5627, -0.2923, -0.0220, -0.2517],
        [-0.8203, -0.5690, -0.3177, -0.0663, -0.3150]])
linear.bias
tensor([-0.1107, -0.0297,  0.0513])


'''







