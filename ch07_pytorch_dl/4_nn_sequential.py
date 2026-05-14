import torch
import torch.nn as nn
from astroid import Lambda
from torchsummary import  summary

# 1.定义数据 random normal - 从标准正态分布中采样随机数

'''


🎲 什么是"标准正态分布"？
标准正态分布（Standard Normal Distribution）是一种概率分布：
均值（mean） = 0
标准差（std） = 1
记作：N(0, 1)


概率密度曲线:
      
      |
  0.4 |        ***
      |       *   *
  0.3 |      *     *
      |     *       *
  0.2 |    *         *
      |   *           *
  0.1 |  *             *
      | *               *
  0.0 +---------------------→
     -3  -2  -1   0   1   2   3
     
大部分值落在 [-3, 3] 区间内
均值在 0 附近

x = torch.randn(10000, 3)  # 生成更多样本以便观察

print(f"均值: {x.mean():.4f}")      # 接近 0
print(f"标准差: {x.std():.4f}")    # 接近 1
print(f"最小值: {x.min():.4f}")    # 约 -3~ -4
print(f"最大值: {x.max():.4f}")

# 神经网络的权重通常用正态分布初始化
weight = torch.randn(10, 5)  # 10×5的权重矩阵

3️⃣ 添加噪声
# 给数据添加高斯噪声
clean_data = torch.tensor([...])
noisy_data = clean_data + 0.1 * torch.randn_like(clean_data)


# 对比示例
print(torch.randn(5))    # 正态分布: [-0.5, 1.2, -0.8, 0.3, 1.5]
print(torch.rand(5))     # 均匀分布: [0.2, 0.7, 0.1, 0.9, 0.4]
print(torch.randint(0, 10, (5,)))  # 随机整数: [3, 7, 1, 9, 2]

'''



'''
x = torch.randn(100, 3)
    ↑              ↑   ↑
    |              |   └─ 每个样本有 3 个特征（输入维度）
    |              └───── 有 100 个样本（批次大小）
    └──────────────────── 从标准正态分布随机生成

'''
x = torch.randn(100, 3)
print(x.shape)    # torch.Size([100, 3])
print(x.ndim)     # 2 （二维张量，即矩阵）
print(x.size())   # torch.Size([100, 3])



# 定义神经网络模型
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(5, 1)
)
print(summary(model, (100, 3)))


# 步骤3: 查看模型结构
summary(model, (100, 3))  # 传入输入形状

'''
输入 x: (100, 3)
  ↓
Linear(3, 5): (100, 3) → (100, 5)
  ↓
ReLU: (100, 5) → (100, 5)  （激活函数不改变形状）
  ↓
Dropout(0.5): (100, 5) → (100, 5)  （丢弃50%的神经元）
  ↓
Linear(5, 1): (100, 5) → (100, 1)
  ↓
输出: (100, 1)

'''


# 定义神经网络模型
model2 = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)


# 定义一个参数初始化函数
def init_parameters(m):
    """
    自定义的模型参数初始化函数
    
    这个函数会被 model.apply() 自动调用，遍历模型中的每一个模块（Module）
    对于 Linear 层，使用特定的初始化策略
    
    参数:
        m (nn.Module): 模型中的一个模块（可能是 Linear、Conv2d、ReLU 等）
    
    初始化策略:
        - 权重 (weight): 使用 Xavier 均匀分布初始化
        - 偏置 (bias): 初始化为 0
    
    为什么要初始化？
        1. 避免梯度消失/爆炸：合适的初始化可以让梯度在反向传播时保持稳定
        2. 加速收敛：好的初始值能让模型更快找到最优解
        3. 打破对称性：随机初始化确保不同神经元学习不同的特征
    
    Xavier 初始化（也叫 Glorot 初始化）:
        - 适用于 Tanh、Sigmoid 等饱和激活函数
        - 从均匀分布 U(-a, a) 中采样，其中 a = sqrt(6 / (fan_in + fan_out))
        - fan_in = 输入维度，fan_out = 输出维度
        - 保持前向和反向传播的方差一致
    
    常见初始化方法对比:
        ┌─────────────────┬──────────────┬────────────────┐
        │   方法           │   适用激活函数 │   特点          │
        ├─────────────────┼──────────────┼────────────────┤
        │ Xavier Uniform  │ Tanh/Sigmoid │ 均匀分布        │
        │ Xavier Normal   │ Tanh/Sigmoid │ 正态分布        │
        │ Kaiming Uniform │ ReLU         │ 适合ReLU系列    │
        │ Kaiming Normal  │ ReLU         │ 适合ReLU系列    │
        │ zeros           │ 偏置         │ 初始化为0       │
        │ ones            │ 偏置         │ 初始化为1       │
        └─────────────────┴──────────────┴────────────────┘
    
    示例:
        # 定义模型
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # 应用初始化
        model.apply(init_parameters)
        
        # 现在所有 Linear 层的权重都用 Xavier 初始化了
    """
    # 判断当前模块是否是线性层（全连接层）
    # isinstance 检查 m 是否为 nn.Linear 类型或其子类
    if isinstance(m, nn.Linear):
        # 对权重使用 Xavier 均匀分布初始化
        # xavier_uniform_ 是就地操作（in-place），直接修改 m.weight
        # 下划线 _ 表示就地修改，不返回新张量
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)  # 偏置初始化为 0.1

        # 如果想用其他初始化方式：
        # nn.init.constant_(m.bias, 0.1)  # 偏置初始化为 0.1
        # nn.init.kaiming_uniform_(m.weight)  # Kaiming 初始化（适合 ReLU）
        # nn.init.normal_(m.weight, mean=0, std=0.01)  # 正态分布初始化

# 3.参数初始化
# apply() 方法会递归地遍历模型中的所有子模块
# 对每个模块调用 init_parameters 函数
# 这样就能统一初始化所有 Linear 层的参数
model2.apply(init_parameters)

# 4.前向传播
output = model2(x)

print(output)

# 验证初始化是否成功
print("\n初始化后的模型参数:")
for name, param in model2.named_parameters():
    print(f"{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
