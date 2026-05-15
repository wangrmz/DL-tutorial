# 导入需要的模块
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.pipeline import Pipeline  # 管道
from sklearn.impute import SimpleImputer  # 处理缺省值
from sklearn.compose import ColumnTransformer  # 列转换器
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 标准化，独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集


# 创建数据集
def create_dataset():
    # 1. 从文件中读取数据
    data = pd.read_csv("../data/house_prices.csv")
    # 2. 处理数据
    data.drop(["Id"], axis=1, inplace=True)
    # 3. 划分特征和目标
    X = data.drop(["SalePrice"], axis=1)
    y = data["SalePrice"]  # 标签
    # 4.划分数据集，测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # return x_train, x_test, y_train, y_test
    # 5.特征工程（特征转换）
    # 5.1 按照特征数据类型分成数值型和类别型
    num_features = x_train.select_dtypes(exclude=['object']).columns
    cat_features = x_train.select_dtypes(include=['object']).columns
    # 5.2定义列转换器
    # 5.2.1 数值型特征：用平均值填充缺失项
    num_transform = Pipeline(steps=[
        ('fillna', SimpleImputer(strategy='mean')),
        ('std', StandardScaler())
    ])
    # 5.2.2 类别型特征：用默认值填充缺失项，在做独热编码处理
    cat_transform = Pipeline(steps=[
        ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # 5.2.3 组合转换器
    transform = ColumnTransformer(transformers=[
        ('num', num_transform, num_features),
        ('cat', cat_transform, cat_features)
    ])
    # 5.3 进行特征转换，构建新的列，组成最终的数据集
    x_train = transform.fit_transform(x_train)
    x_test = transform.transform(x_test)
    # 稀疏矩阵的处理，转稠密矩阵，列名重新生成，并且重构构建成DataFrame
    x_train = pd.DataFrame(x_train.toarray(), columns=transform.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=transform.get_feature_names_out())
    # 6. 构建tensor数据集
    # DataFrame需要转成ndarray
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    # 返回训练集和测试集，以及特征的数量
    return train_dataset, test_dataset, x_train.shape[1]


# 测试
# 1.加载数据
train_dataset, test_dataset, n_features = create_dataset()
print(n_features)

# 2. 搭建模型 使用Sequential进行搭建
# 输入层，提取的本质特征，隐藏层，输出层
model = nn.Sequential(nn.Linear(n_features, 128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(128, 1),
                      )


# 3.自定义损失函数 基于MSE
def log_rmse(pred, target):
    """
    计算对数均方根误差（Log RMSE）
    
    为什么要用 Log RMSE？
        - 房价数据通常呈偏态分布（有少数非常高的房价）
        - 直接预测原始值会受到极端值的影响
        - 取对数后可以缩小数值范围，使分布更接近正态分布
        - 这样模型更关注相对误差而不是绝对误差
        
    例如:
        真实房价: 100万, 预测: 110万 → 绝对误差10万，相对误差10%
        真实房价: 10万,  预测: 20万  → 绝对误差10万，相对误差100%
        
        如果直接用MSE，两个样本的误差相同
        但用Log RMSE，第二个样本的误差会更大（因为相对误差大）
    
    参数:
        pred: 模型预测值
        target: 真实标签
    
    返回:
        Log RMSE 损失值
    """
    mse = nn.MSELoss()
    # squeeze_() 移除维度为1的维度，例如 (batch_size, 1) → (batch_size,)
    pred = pred.squeeze()
    # clamp 限制预测值在 [1, +∞) 范围内，避免 log(0) 或 log(负数)
    pred = torch.clamp(pred, 1, float("inf"))
    # 计算 log(pred) 和 log(target) 的 MSE，然后开平方
    return torch.sqrt(mse(torch.log(pred), torch.log(target)))


# 4.模型训练和测试
def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size):
    """
    训练和测试模型
    
    参数:
        model: PyTorch 模型
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        lr: 学习率
        epoch_num: 训练轮数
        batch_size: 批次大小
    
    返回:
        train_loss_list: 每轮的训缅损失列表
        test_loss_list: 每轮的测试损失列表
    """
    # 1.初始化相关操作
    def init_params(layer):
        """
        自定义参数初始化函数
        
        只对 Linear 层进行 Xavier 初始化
        BatchNorm、Dropout、ReLU 等层不需要手动初始化
        
        为什么只初始化 Linear 层？
            - Xavier 初始化要求张量至少是2维的（fan_in 和 fan_out）
            - BatchNorm 的 weight 是1维的，不能用 Xavier
            - BatchNorm 有自己的初始化策略（weight=1, bias=0）
            - Dropout 和 ReLU 没有可学习参数
        """
        # 判断是否是线性层（全连接层）
        if isinstance(layer, nn.Linear):
            # 对 Linear 层的权重使用 Xavier 正态分布初始化
            # Xavier 初始化适合 Tanh/Sigmoid，但对于 ReLU 也能用
            nn.init.xavier_normal_(layer.weight)
            # 偏置初始化为0
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    # 1.1 参数初始化
    # apply() 会递归遍历模型的所有子模块，对每个模块调用 init_params
    model.apply(init_params)

    # 定义优化器
    # Adam 优化器结合了 Momentum 和 RMSProp 的优点
    # - 自适应学习率：每个参数有自己的学习率
    # - 动量加速：考虑历史梯度方向
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 定义训练误差和测试误差变化列表
    train_loss_list = []
    test_loss_list = []

    # 2. 模型训练
    for epoch in range(epoch_num):
        # train() 模式：启用 Dropout 和 BatchNorm 的训练行为
        # - Dropout: 随机丢弃神经元
        # - BatchNorm: 使用当前批次的统计量（mean/var）
        model.train()
        
        # 2.1 创建 DataLoader
        # shuffle=True: 每个epoch打乱数据顺序，增加随机性，防止过拟合
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss_total = 0
        
        # 2.2 按照批次迭代训练模型
        for batch_idx, (X, y) in enumerate(train_loader):
            # 前向传播：计算预测值
            y_pred = model(X)
            
            # 计算损失：使用 Log RMSE
            # squeeze() 移除多余的维度，确保 y_pred 和 y 形状一致
            loss_value = log_rmse(y_pred.squeeze(), y)
            
            # 反向传播：计算梯度
            loss_value.backward()
            
            # 更新参数：根据梯度和学习率调整权重
            optimizer.step()
            
            # 梯度清零：PyTorch 默认累加梯度，需要手动清零
            # 必须在 step() 之后清零，否则会影响下一次更新
            optimizer.zero_grad()
            
            # 累加损失：乘以样本数得到总损失
            train_loss_total += loss_value.item() * X.shape[0]
        
        # 计算平均训练损失
        this_train_loss = train_loss_total / len(train_dataset)
        train_loss_list.append(this_train_loss)

        # 3.测试
        # eval() 模式：关闭 Dropout，使用 BatchNorm 的运行统计量
        # - Dropout: 不丢弃神经元，所有神经元都参与
        # - BatchNorm: 使用训练阶段积累的 running_mean 和 running_var
        model.eval()
        
        # 3.1 定义测试数据集的 DataLoader
        # shuffle=False: 测试时不需要打乱数据
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 3.2 计算测试误差
        test_loss_total = 0
        
        # no_grad(): 测试时不需要计算梯度，节省内存和计算资源
        with torch.no_grad():
            for X, y in test_loader:
                # 前向传播
                y_pred = model(X)
                # 计算损失
                loss_value = log_rmse(y_pred.squeeze(), y)
                # 累加损失
                test_loss_total += loss_value.item() * X.shape[0]
        
        # 计算平均测试损失
        this_test_loss = test_loss_total / len(test_dataset)
        test_loss_list.append(this_test_loss)
        
        # 打印当前 epoch 的损失
        print(f"Epoch: {epoch + 1}/{epoch_num}, Train Loss: {this_train_loss:.4f}, Test Loss: {this_test_loss:.4f}")

    return train_loss_list, test_loss_list


# 超参数
lr = 0.01
epoch_num = 200
batch_size = 64
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size)


# 画图
plt.plot(train_loss_list, 'r-',label='Train Loss',linewidth = 2)
plt.plot( test_loss_list, 'k-',label='Test Loss',linewidth = 2)
plt.show()


'''

1. 为什么要用 Log RMSE？
房价数据特点:
- 大部分房子: 50-200万
- 少数豪宅: 1000万+

直接用MSE的问题:
- 预测100万的房子误差10万 → MSE = 100亿
- 预测1000万的房子误差10万 → MSE = 100亿
- 两个误差相同，但相对误差完全不同！

Log RMSE的优势:
- log(100万) ≈ 13.8, log(110万) ≈ 13.9 → 误差小
- log(10万) ≈ 11.5, log(20万) ≈ 12.2 → 误差大
- 更关注相对误差，适合偏态分布的数据

# 测试时不需要梯度，用no_grad可以：
# 1. 节省内存（不保存中间变量）
# 2. 加速计算（不做梯度追踪）
with torch.no_grad():
    y_pred = model(X)  # 不会构建计算图


# 测试时不需要梯度，用no_grad可以：
# 1. 节省内存（不保存中间变量）
# 2. 加速计算（不做梯度追踪）
with torch.no_grad():
    y_pred = model(X)  # 不会构建计算图



'''

'''
1. 参数初始化
   ↓
2. 创建优化器 (Adam)
   ↓
3. 循环 epoch_num 次:
   ├─ 训练阶段 (model.train())
   │  ├─ 打乱数据 (shuffle=True)
   │  ├─ 分批迭代
   │  │  ├─ 前向传播
   │  │  ├─ 计算损失
   │  │  ├─ 反向传播
   │  │  ├─ 更新参数
   │  │  └─ 清零梯度
   │  └─ 记录训练损失
   │
   └─ 测试阶段 (model.eval())
      ├─ 不打乱数据 (shuffle=False)
      ├─ no_grad() 节省资源
      ├─ 分批迭代
      │  ├─ 前向传播
      │  └─ 计算损失
      └─ 记录测试损失

'''


