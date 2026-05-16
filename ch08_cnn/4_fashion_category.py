import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

'''
数据在csv文件中，每行数据为1个样本，第1列为标签，2到785列为784个像素。我们需要将其转换为28×28的形状：
'''

# 读取数据
fashion_mnist_train = pd.read_csv("../data/fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("../data/fashion-mnist_test.csv")
# 将数据转换为张量，原数据形状为n×1×784，转换为n×1×28×28的张量
X_train = torch.tensor(fashion_mnist_train.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(fashion_mnist_train.iloc[:, 0].values, dtype=torch.int64)
X_test = torch.tensor(fashion_mnist_test.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test = torch.tensor(fashion_mnist_test.iloc[:, 0].values, dtype=torch.int64)
# 灰度
plt.imshow(X_train[12345, 0, :, :], cmap="gray")
plt.show()

# 构建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 搭建模型
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),  # 拉平
    nn.Linear(400, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)

# 查看各层输出数据的形状
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in model:
    X = layer(X)
    print(f"{layer.__class__.__name__:<12}output shape: {X.shape}")

'''

先初始化线性层和卷积层的权重参数。使用交叉熵损失函数和SGD优化方法。
每个epoch中在训练集上训练模型，并在验证集上验证模型的准确率


'''


# 模型训练
def train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    """
    训练和验证模型
    
    参数:
        model: PyTorch 模型
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        lr: 学习率
        epoch_num: 训练轮数
        batch_size: 批次大小
        device: 计算设备 (cpu/mps/cuda)
    """
    def init_weights(layer):
        """对线性层和卷积层使用Xavier均匀分布初始化参数"""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight)

    model.apply(init_weights)  # 初始化参数
    model.to(device)  # 将模型加载到设备
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器
    
    for epoch in range(epoch_num):
        # ========== 训练过程 ==========
        model.train()  # 将模型设置为训练模式
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        loss_accumulate = 0
        train_correct_accumulate = 0
        
        for batch_count, (X, y) in enumerate(train_loader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            
            # 计算损失
            loss_value = loss_fn(output, y)
            
            # 反向传播
            optimizer.zero_grad() # 清空梯度
            loss_value.backward() # 反向传播
            optimizer.step() # 更新参数
            
            # 累加损失
            loss_accumulate += loss_value.item()
            
            # 累加正确输出的数量
            _, pred = output.max(1)
            train_correct_accumulate += pred.eq(y).sum().item()
            
            # 打印进度条
            progress = int((batch_count + 1) / len(train_loader) * 50)
            print(f"\repoch:{epoch+1:0>2}/{epoch_num}[{'=' * progress:<50}]", end="")
        
        this_loss = loss_accumulate / len(train_loader)  # 计算平均损失
        this_train_correct = train_correct_accumulate / len(train_dataset)  # 计算训练准确率

        # ========== 验证过程 ==========
        model.eval()  # 将模型设置为评估模式
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        test_correct_accumulate = 0
        
        with torch.no_grad():  # 关闭梯度计算
            for X, y in test_loader:
                # 前向传播
                X, y = X.to(device), y.to(device)
                output = model(X)
                
                # 累加正确输出的数量
                _, pred = output.max(1)
                test_correct_accumulate += pred.eq(y).sum().item()
        
        this_test_correct = test_correct_accumulate / len(test_dataset)  # 计算验证准确率

        # 打印损失，训练准确率，验证准确率
        print(f" loss:{this_loss:.6f}, train_acc:{this_train_correct:.4f}, test_acc:{this_test_correct:.4f}")

# 设置计算设备：优先使用 MPS (macOS)，其次 CUDA，最后 CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS Apple Silicon
    print("使用 MPS 设备 (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
    print("使用 CUDA 设备")
else:
    device = torch.device("cpu")  # CPU
    print("使用 CPU 设备")

# 开始训练
train(model, train_dataset, test_dataset, lr=0.1, epoch_num=20, batch_size=256, device=device)


