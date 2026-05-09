# ReLU激活函数层
from common.functions import sigmoid,softmax,cross_entropy
import numpy as np

class ReLU:
    """
    ReLU (Rectified Linear Unit) 激活函数层
    
    ReLU函数定义:
        f(x) = max(0, x)
        - 如果 x > 0, 输出 x
        - 如果 x <= 0, 输出 0
    
    特点:
        - 计算简单高效
        - 缓解梯度消失问题（相比sigmoid）
        - 使神经网络具有稀疏性（部分神经元输出为0）
        - 是目前最常用的激活函数之一
    
    导数:
        f'(x) = 1 (当 x > 0)
        f'(x) = 0 (当 x <= 0)
    """

    # 初始化
    def __init__(self):
        """
        初始化ReLU层
        
        属性:
            self.mask: 布尔数组，记录前向传播时哪些位置的输入x <= 0
                      形状与输入x相同
                      用于反向传播时快速定位需要置零的位置
        
        为什么要保存mask?
            - 反向传播时需要知道前向传播时哪些位置被"关闭"了(x<=0)
            - 这些位置的梯度应该为0（因为ReLU在x<=0时导数为0）
            - 保存mask可以避免重复计算，提高效率
        """
        # 内部属性，记录那些x<0的位置
        self.mask = None

    # 前向传播
    def forward(self, x):
        """
        ReLU前向传播：将负值置为0
        
        参数:
            x (numpy.ndarray): 输入数据，可以是任意形状
        
        返回:
            numpy.ndarray: ReLU激活后的输出，形状与输入相同
                          所有负值都被替换为0
        
        计算公式:
            y = max(0, x)
        
        示例:
            输入:  [-2, -1, 0, 1, 2]
            输出:  [ 0,  0, 0, 1, 2]
        
        实现步骤:
            1. 创建布尔掩码，标记x <= 0的位置
            2. 复制输入数据（避免修改原始数据）
            3. 将标记位置的元素设为0
        """
        # 记录小于等于0的值的位置（True表示该位置x<=0）
        self.mask = (x <= 0)

        # 复制输入数据，避免直接修改原始输入
        y = x.copy()

        # 将所有x<=0的位置的值改为0
        y[self.mask] = 0

        return y

    # 反向传播
    def backward(self, dout):
        """
        ReLU反向传播：根据链式法则传递梯度
        
        参数:
            dout (numpy.ndarray): 上游传来的梯度，形状与前向输出相同
                                 表示损失函数对ReLU输出的偏导数 ∂L/∂y
        
        返回:
            numpy.ndarray: 传递给下一层的梯度，形状与输入相同
                          表示损失函数对ReLU输入的偏导数 ∂L/∂x
        
        链式法则:
            ∂L/∂x = ∂L/∂y · ∂y/∂x
            
            其中 ∂y/∂x 是ReLU的导数:
                - 如果 x > 0:  ∂y/∂x = 1  →  ∂L/∂x = ∂L/∂y · 1 = dout
                - 如果 x <= 0: ∂y/∂x = 0  →  ∂L/∂x = ∂L/∂y · 0 = 0
        
        直观理解:
            - 前向传播时被"关闭"的神经元(x<=0)，反向传播时梯度也为0
            - 前向传播时"激活"的神经元(x>0)，梯度原样传递
            - 这就像一扇"门"：前向时关上的门，反向时也不让梯度通过
        
        示例:
            前向输入:  [-2, -1, 0, 1, 2]
            前向输出:  [ 0,  0, 0, 1, 2]
            mask:      [ T,  T, T, F, F]  ← 记录哪些位置<=0
            
            上游梯度:  [0.1, 0.2, 0.3, 0.4, 0.5]
            下游梯度:  [  0,   0,   0, 0.4, 0.5]  ← mask位置梯度变为0
        
        实现步骤:
            1. 复制上游梯度
            2. 将前向时x<=0位置的梯度设为0
        """
        # 复制上游传来的梯度
        dx = dout.copy()

        # 将前向传播时x<=0的位置的梯度设为0
        # 因为这些位置在前向时被"关闭"了，反向时梯度也不能通过
        dx[self.mask] = 0

        return dx


# Sigmoid激活函数层
class Sigmoid:
    def __init__(self):
        # 定义内部属性，记录输出值y,用于反向传播时计算梯度
        self.y = None

    # 前向传播
    def forward(self, x):
        """
        Sigmoid前向传播：将输入映射到0-1区间

        参数:
            x (numpy.ndarray): 输入数据，可以是任意形状

        返回:
            numpy.ndarray: Sigmoid激活后的输出，形状与输入相同
                          所有输入映射到0-1区间

        计算公式:
            y = 1 / (1 + exp(-x))
        """
        y = sigmoid(x)
        self.y = y
        return y

    # 反向传播
    def backward(self, dout):
        """
        Sigmoid反向传播：根据链式法则传递梯度
        参数:
            dout (numpy.ndarray): 上游传来的梯度，形状与前向输出相同
                                 表示损失函数对Sigmoid输出的偏导数 ∂L/∂y
        返回:
            numpy.ndarray: 传递给下一层的梯度，形状与输入相同
                          表示损失函数对Sigmoid输入的偏导数 ∂L/∂x
        链式法则:
            ∂L/∂x = ∂L/∂y · ∂y
        """
        dx = dout * (1.0 - self.y) * self.y





# 创建一个全连接层
class Affine:
    def __init__(self, W, b):
        """
        创建一个全连接层
        参数:
            W (numpy.ndarray): 权重矩阵，形状为 (输入维度, 输出维度)
            b (numpy.ndarray): 偏置向量，形状为 (输出维度,)
        """
        self.W = W
        self.b = b
        self.X = None
        self.originl_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, X):
        self.originl_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(self.X, self.W) + self.b
        return y

    #  反向传播
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        dX = dX.reshape(*self.originl_x_shape)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX

'''
原始标签: [猫, 鸟, 狗, 猫]

【顺序编码】(Sequential/Categorical)
t = [0, 2, 1, 0]
形状: (4,)
含义: 直接用数字表示类别

【One-hot编码】(One-Hot)
t = [[1, 0, 0],   # 猫 → 第0位是1
     [0, 0, 1],   # 鸟 → 第2位是1
     [0, 1, 0],   # 狗 → 第1位是1
     [1, 0, 0]]   # 猫 → 第0位是1
形状: (4, 3)
含义: 用向量表示,对应位置为1

'''
# 输出层：Softmax + Cross Entropy Loss
class SoftmaxWithLoss:
    """
    Softmax激活函数 + 交叉熵损失函数的组合层
    
    这是多分类神经网络的标准输出层配置:
        - Softmax: 将原始输出转换为概率分布
        - Cross Entropy: 计算预测概率与真实标签的差异
    
    为什么要把它们放在一起?
        1. 数值稳定性更好
        2. 反向传播的梯度公式更简洁: dx = (y - t) / batch_size
        3. 避免单独计算softmax和cross entropy时的精度损失
    
    数学原理:
        前向: L = -Σ t_k · log(y_k)
        反向: ∂L/∂x = (y - t) / N
        其中 y=softmax(x), t是真实标签, N是批量大小
    
    这个简洁的梯度公式是Softmax+CrossEntropy组合的重要优势!
    """
    
    # 初始化
    def __init__(self):
        """
        初始化SoftmaxWithLoss层
        
        属性:
            self.loss: 计算得到的损失值（标量）
            self.y: Softmax的输出，即预测的概率分布
                   形状: (batch_size, num_classes)
                   每行的和为1，表示每个类别的概率
            self.t: 监督数据（真实标签）
                   可以是两种格式:
                   - one-hot编码: 形状 (batch_size, num_classes)
                   - 顺序编码: 形状 (batch_size,)，值为类别索引
        """
        self.loss = None
        self.y = None  # softmax的输出，预测值
        self.t = None  # 监督数据，真实值

    # 前向传播
    def forward(self, x, t):
        """
        执行前向传播：计算Softmax输出和交叉熵损失
        
        参数:
            x (numpy.ndarray): 网络的原始输出（logits），形状 (N, K)
                              N = batch_size（批量大小）
                              K = num_classes（类别数量）
                              这些值可以是任意实数，不需要在0-1之间
            
            t (numpy.ndarray): 真实标签，有两种可能的格式:
                              格式1 - One-hot编码: 形状 (N, K)
                                      例如: [[0,1,0], [0,0,1]] 表示第1类和第2类
                              格式2 - 顺序编码: 形状 (N,)
                                      例如: [1, 2] 表示第1类和第2类
        
        返回:
            float: 交叉熵损失值，越小表示预测越准确
        
        前向传播流程:
            1. 保存真实标签 t
            2. 对输入 x 应用 Softmax，得到概率分布 y
            3. 计算 y 和 t 之间的交叉熵损失
            4. 返回损失值
        
        示例:
            输入 x: [[2.0, 1.0, 0.1],   # 网络原始输出
                     [0.5, 2.5, 1.0]]
            
            Softmax后 y: [[0.659, 0.242, 0.099],   # 转换为概率
                          [0.187, 0.619, 0.194]]
            
            真实标签 t (one-hot): [[0, 1, 0],     # 第1个样本是类别1
                                   [0, 0, 1]]     # 第2个样本是类别2
            
            损失 L = -(log(0.242) + log(0.194)) / 2 ≈ 1.15
        """
        self.t = t
        self.y = softmax(x)  # 将原始输出转换为概率分布
        self.loss = cross_entropy(self.y, self.t)  # 计算交叉熵损失
        return self.loss

    # 反向传播
    def backward(self, dout=1):
        """
        执行反向传播：计算损失对输入的梯度
        
        参数:
            dout (float): 上游传来的梯度，默认为1
                         因为这是输出层，通常没有上游，所以设为1
        
        返回:
            numpy.ndarray: 损失对输入x的梯度，形状 (N, K)
                          这个梯度会传递给前一层的Affine层
        
        ⭐ 核心公式（非常重要！）:
            ∂L/∂x = (y - t) / batch_size
        
        这个公式的推导:
            L = -Σ t_k · log(y_k)  （交叉熵）
            y_k = exp(x_k) / Σ exp(x_j)  （softmax）
            
            通过链式法则推导可得:
            ∂L/∂x_k = y_k - t_k
            
            再除以batch_size是为了求平均梯度
        
        为什么这么简洁？
            - Softmax和CrossEntropy的导数相互抵消，得到简单形式
            - 这就是为什么要将它们放在一层的原因！
            - 相比分别计算，这样既高效又稳定
        
        直观理解:
            - y 是预测概率，t 是真实标签
            - y - t 表示预测误差
            - 如果预测正确(y≈t)，梯度接近0
            - 如果预测错误(y≠t)，梯度较大，需要大幅调整
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        关于标签格式的说明:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        【One-hot编码 vs 顺序编码】
        
        假设我们有3个类别(0, 1, 2)，批量大小为4:
        
        顺序编码 (Sequential/Categorical encoding):
            t = [0, 2, 1, 0]
            含义: 第1个样本是类别0
                 第2个样本是类别2
                 第3个样本是类别1
                 第4个样本是类别0
            形状: (4,) - 一维数组
            优点: 节省内存，直观
        
        One-hot编码 (One-Hot encoding):
            t = [[1, 0, 0],   # 类别0 → [1,0,0]
                 [0, 0, 1],   # 类别2 → [0,0,1]
                 [0, 1, 0],   # 类别1 → [0,1,0]
                 [1, 0, 0]]   # 类别0 → [1,0,0]
            形状: (4, 3) - 二维数组
            优点: 便于矩阵运算，某些框架要求
        
        【转换方法】:
        
        顺序编码 → One-hot编码:
            import numpy as np
            t_sequential = np.array([0, 2, 1, 0])
            num_classes = 3
            t_onehot = np.eye(num_classes)[t_sequential]
            # 结果: [[1,0,0], [0,0,1], [0,1,0], [1,0,0]]
        
        One-hot编码 → 顺序编码:
            t_onehot = np.array([[1,0,0], [0,0,1], [0,1,0], [1,0,0]])
            t_sequential = np.argmax(t_onehot, axis=1)
            # 结果: [0, 2, 1, 0]
        
        【代码中的处理】:
        
        情况1: 如果 t 是 one-hot 编码 (t.size == y.size)
            直接使用公式: dx = (y - t) / batch_size
            
            示例:
                y = [[0.7, 0.2, 0.1],   # 预测概率
                     [0.1, 0.2, 0.7]]
                t = [[1, 0, 0],          # one-hot标签
                     [0, 0, 1]]
                
                y - t = [[-0.3, 0.2, 0.1],   # 预测误差
                         [0.1, 0.2, -0.3]]
                
                dx = (y - t) / 2  # 除以batch_size
        
        情况2: 如果 t 是顺序编码 (t.size != y.size)
            需要先将顺序编码转换为one-hot的效果，然后相减
            
            步骤:
                1. 复制 y 得到 dx
                2. 对于每个样本，将其真实类别对应的概率减1
                3. 除以 batch_size
            
            示例:
                y = [[0.7, 0.2, 0.1],   # 预测概率
                     [0.1, 0.2, 0.7]]
                t = [0, 2]               # 顺序编码
                
                dx = y.copy() = [[0.7, 0.2, 0.1],
                                 [0.1, 0.2, 0.7]]
                
                # 第0个样本的真实类别是0，所以dx[0, 0] -= 1
                # 第1个样本的真实类别是2，所以dx[1, 2] -= 1
                dx[np.arange(2), t] -= 1
                
                dx = [[-0.3, 0.2, 0.1],   # 0.7-1=-0.3
                      [0.1, 0.2, -0.3]]   # 0.7-1=-0.3
                
                dx = dx / 2  # 除以batch_size
            
            注意: 两种方式得到的结果完全相同！
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        batch_size = self.t.shape[0]  # 获取批量大小
        
        # 判断标签格式：如果是独热编码的标签，就直接代入公式计算
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            # One-hot编码格式: 直接相减
            # 例如: y=[0.7,0.2,0.1], t=[1,0,0] → dx=[-0.3,0.2,0.1]
            dx = (self.y - self.t) / batch_size
        
        # 如果是顺序编码的标签，就需要找到分类号对应的值，然后相减
        else:
            # 顺序编码格式: 先复制y，再将真实类别位置的值减1
            dx = self.y.copy()
            
            # np.arange(batch_size) 生成 [0, 1, 2, ..., batch_size-1]
            # self.t 是真实标签数组，如 [0, 2, 1, 0]
            # dx[np.arange(batch_size), self.t] 选择:
            #   dx[0, t[0]], dx[1, t[1]], dx[2, t[2]], ...
            # 然后将这些位置的值减1
            #
            # 示例:
            #   dx = [[0.7, 0.2, 0.1],
            #         [0.1, 0.2, 0.7]]
            #   t = [0, 2]
            #   np.arange(2) = [0, 1]
            #   dx[[0,1], [0,2]] 选择 dx[0,0]和dx[1,2]
            #   dx[0,0] -= 1 → 0.7-1 = -0.3
            #   dx[1,2] -= 1 → 0.7-1 = -0.3
            dx[np.arange(batch_size), self.t] -= 1
            
            # 除以batch_size，求平均梯度
            dx = dx / batch_size

        return dx




















