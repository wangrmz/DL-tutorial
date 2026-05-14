'''
在神经网络框架中，由多个层组成的组件称之为 模块（Module）。
在PyTorch中模型就是一个Module，各网络层、模块也是Module。Module是所有神经网络的基类。

在定义一个Module时，我们需要继承torch.nn.Module并主要实现两个方法：
__init__：定义网络各层的结构，并初始化参数。
forward：根据输入进行前向传播，并返回输出。计算其输出关于输入的梯度，可通过其反向传播函数进行访问（通常自动发生）。forward方法是每次调用的具体实现。
第1个隐藏层：使用Xavier正态分布初始化权重，激活函数使用Tanh。
第2个隐藏层：使用He正态分布初始化权重，激活函数使用ReLU。
输出层：按默认方式初始化，激活函数使用Softmax。

'''

import torch
import torch.nn as nn
from torchsummary import  summary



class Model(nn.Module):
    # 初始化 主要是层级关系 三层
    def __init__(self):
        super(Model, self).__init__()  # 调用父类初始化
        # 定义三个线性层
        self.linear1 = nn.Linear(3, 4)  # 第1个隐藏层，3个输入，4个输出
        # 使用xavier正态分布初始化参数
        nn.init.xavier_normal_(self.linear1.weight)  # 初始化权重参数
        self.linear2 = nn.Linear(4, 4)  # 第2个隐藏层，4个输入，4个输出
        # 使用kaiming正态分布初始化参数
        nn.init.kaiming_normal_(self.linear2.weight)  # 初始化权重参数
        self.out = nn.Linear(4, 2)  # 输出层，4个输入，2个输出，默认使用He均匀分布初始化

    # 前向传播
    def forward(self, x):
        # 前一层的输出是后一层的输入
        x = self.linear1(x)  # 经过第1个隐藏层
        x = torch.tanh(x)  # 激活函数
        x = self.linear2(x)  # 经过第2个隐藏层
        x = torch.relu(x)  # 激活函数
        x = self.out(x)  # 经过输出层
        x = torch.softmax(x, dim=1)  # 输出层的激活函数使用softmax
        return x


# 创建神经网络模型
model = Model()
# 定义输入数据获取模型的输出
output = model(torch.randn(10, 3))
print("输出：\n", output)
print()

# 1.可以单独查看
print("第1个隐藏层的参数：")
print("第1个隐藏层的参数：\n", model.linear1.weight)
print("第1个隐藏层的参数：\n", model.linear1.bias)
print()
print("第2个隐藏层的参数：")
print("第2个隐藏层的参数：\n", model.linear2.weight)
print("第2个隐藏层的参数：\n", model.linear2.bias)
print()
print("输出层的参数：")
print("输出层的参数：\n", model.out.weight)
print("输出层的参数：\n", model.out.bias)
print()

# 2.使用named_parameters()查看各层参数
print("模型参数：")
for name, param in model.named_parameters():
    print(name, param)
    print()

# 3.使用state_dict()查看各层参数
print("模型参数：\n", model.state_dict())


# 可使用torchsummary.summary来查看模型结构与参数数量
print('torchsummary.summary来查看模型结构与参数数量')
summary(model, (3,),batch_size=10 )



'''
torchsummary.summary来查看模型结构与参数数量
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                    [10, 4]              16
            Linear-2                    [10, 4]              20
            Linear-3                    [10, 2]              10
================================================================
Total params: 46
Trainable params: 46
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
'''

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                    [10, 4]              16
            Linear-2                    [10, 4]              20
            Linear-3                    [10, 2]              10
================================================================
Total params: 46
Trainable params: 46
Non-trainable params: 0
----------------------------------------------------------------


参数量 = 输入维度 × 输出维度 + 输出维度（偏置）
       = in_features × out_features + out_features
       = (in_features + 1) × out_features


self.linear1 = nn.Linear(3, 4)  # 输入3维，输出4维


权重 W: 形状 (3, 4) → 3 × 4 = 12 个参数
偏置 b: 形状 (4,)   → 4 个参数
总计: 12 + 4 = 16 个参数 ✅


输入: [x1, x2, x3]  (3个特征)
      ↓
权重矩阵 W (3×4):
     ┌           ┐
     │ w11 w12 w13 w14 │
     │ w21 w22 w23 w24 │
     │ w31 w32 w33 w34 │
     └           ┘
     
偏置 b (4个):
[b1, b2, b3, b4]

总参数: 12个权重 + 4个偏置 = 16个


self.linear2 = nn.Linear(4, 4)  # 输入4维，输出4维


权重 W: 形状 (4, 4) → 4 × 4 = 16 个参数
偏置 b: 形状 (4,)   → 4 个参数
总计: 16 + 4 = 20 个参数 ✅

self.out = nn.Linear(4, 2)  # 输入4维，输出2维

权重 W: 形状 (4, 2) → 4 × 2 = 8 个参数
偏置 b: 形状 (2,)   → 2 个参数
总计: 8 + 2 = 10 个参数 ✅


Total params = 16 + 20 + 10 = 46 ✅


Output Shape: [10, 4]
              ↑   ↑
              |   └─ 输出维度（神经元数量）
              └───── batch_size（批次大小）


summary(model, (3,), batch_size=10)
#                  ↑          ↑
#              输入维度    批次大小

所以每层的输出形状都是 [10, 输出维度]：
Linear-1: [10, 4] - 10个样本，每个样本4维输出
Linear-2: [10, 4] - 10个样本，每个样本4维输出
Linear-3: [10, 2] - 10个样本，每个样本2维输出（2个类别）

Input size (MB): 0.00              # 输入数据占用的内存
Forward/backward pass size (MB): 0.00  # 前向/反向传播时的中间变量内存
Params size (MB): 0.00             # 参数占用的内存
Estimated Total Size (MB): 0.00    # 预估总内存占用


# 方法1: 直接查看参数形状
print(model.linear1.weight.shape)  # torch.Size([4, 3])
print(model.linear1.bias.shape)    # torch.Size([4])
print(f"Linear1 参数数: {4*3 + 4} = 16")

# 方法2: 统计所有参数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数: {total_params}")  # 输出: 46

# 方法3: 只统计可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_params}")  # 输出: 46


🎯 为什么要查看模型参数？
1. 调试模型结构

# 确认每层的输入输出维度是否正确
# 比如发现某层输出维度不对，可以及时调整

2. 估算模型复杂度
参数量越多 → 模型越复杂 → 可能过拟合
参数量越少 → 模型越简单 → 可能欠拟合

46个参数是非常小的模型，适合简单任务

3. 评估训练资源需求
参数量大 → 需要更多显存、更长的训练时间
参数量小 → 训练快，适合快速实验

4. 对比不同模型
# 可以快速比较不同架构的参数量
model_A: 100万参数
model_B: 1000万参数
→ model_B 更复杂，可能需要更多数据


torchsummary 的核心价值:

1. 一目了然地看到模型结构
2. 自动计算每层的参数数量
3. 帮助评估模型复杂度和资源需求
4. 快速发现维度不匹配的问题

参数计算公式:
Linear(in, out) 的参数 = in × out + out = (in + 1) × out

你的模型:
Linear(3,4): 16参数
Linear(4,4): 20参数  
Linear(4,2): 10参数
总计: 46参数 ← 非常轻量级的模型！


'''