'''

SGD有以下问题：
局部最优解：陷入局部最优，尤其在非凸函数中，难以找到全局最优解。
鞍点：陷入鞍点，梯度为0，导致训练停滞。
收敛速度慢：高维或非凸函数中，收敛速度较慢。
学习率选择：学习率过大导致震荡或不收敛，过小则收敛速度慢。

'''


# SGD 随机梯度下降
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr  # 学习率

    # 更新参数 ，传入参数字典和梯度字典
    def update(self, params, grads):
        # 遍历传入的所有参数，按照公式更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]


import numpy as np


# 动量法优化器 (Momentum)
class Momentum:
    """
    Momentum (动量法) 优化器
    
    核心思想:
        模拟物理学中的动量概念，在梯度下降过程中积累"速度"
        就像球从山上滚下来，会越来越快，并且能冲过小坑洼
    
    解决的问题:
        1. 加速收敛：在梯度方向一致时，速度会累积，加快下降
        2. 抑制震荡：在梯度方向变化时，动量会平滑更新路径
        3. 跳出局部最优：足够的动量可以冲出浅的局部最优解
        4. 穿越鞍点：即使梯度很小，历史动量也能推动参数继续更新
    
    更新公式:
        v = momentum * v - lr * gradient  # 更新速度
        params += v                        # 用速度更新参数
    
    其中:
        - v: 速度向量，记录历史梯度的加权和
        - momentum: 动量因子（通常0.9），控制历史梯度的影响程度
        - lr: 学习率，控制当前梯度的影响程度
    
    直观理解:
        - momentum=0.9 表示保留90%的历史速度，只受10%新梯度影响
        - 就像推一个重球，一旦动起来就不容易停下
        - 如果一直往同一个方向推，速度会越来越快
        - 如果方向频繁改变，速度会被平均化，减少震荡
    
    与SGD的对比:
        SGD:  每次只看当前梯度，容易震荡，收敛慢
        Momentum: 考虑历史梯度，路径更平滑，收敛更快
    
    示例:
        假设在下山过程中:
        - 第1步: 梯度向下，v = 0.9*0 - 0.01*(-1) = 0.01
        - 第2步: 梯度仍向下，v = 0.9*0.01 - 0.01*(-1) = 0.019
        - 第3步: 梯度仍向下，v = 0.9*0.019 - 0.01*(-1) = 0.0271
        → 速度越来越快！
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化Momentum优化器
        
        参数:
            lr (float): 学习率，默认0.01
                       控制每次更新的步长大小
            momentum (float): 动量因子，默认0.9
                             范围通常在[0.5, 0.99]之间
                             值越大，历史梯度影响越大，越平滑
        
        属性:
            self.v (dict): 速度字典，为每个参数维护一个速度向量
                          初始为None，第一次update时初始化
                          形状与对应参数相同，初始值为0
        """
        self.lr = lr  # 学习率
        self.momentum = momentum  # 动量因子
        self.v = None  # 速度(历史负梯度的加权和)

    # 更新参数
    def update(self, params, grads):
        """
        使用动量法更新网络参数
        
        参数:
            params (dict): 参数字典，包含需要更新的权重和偏置
                          例如: {'W1': array(...), 'b1': array(...), ...}
            grads (dict): 梯度字典，包含各参数的梯度
                         例如: {'W1': array(...), 'b1': array(...), ...}
                         形状与params中对应参数相同
        
        更新流程:
            1. 首次调用时，初始化速度字典v（全零）
            2. 对每个参数:
               a. 更新速度: v = momentum * v_old - lr * gradient
               b. 更新参数: param += v
        
        为什么要初始化v?
            - 第一次更新时没有历史速度，需要从0开始
            - 使用np.zeros_like保证v的形状与参数完全一致
            - 只在第一次调用时初始化，之后保持状态
        
        公式详解:
            v_new = momentum * v_old - lr * grad
            param_new = param_old + v_new
            
            展开来看:
            param_new = param_old + momentum * v_old - lr * grad
            
            这意味着:
            - momentum * v_old: 继承之前的运动趋势（惯性）
            - lr * grad: 根据当前梯度调整方向
            - 两者结合，既保持稳定性，又能响应新的梯度信息
        
        示例:
            假设有参数 W，初始速度 v=0
            
            第1次更新:
                grad = 0.5
                v = 0.9 * 0 - 0.01 * 0.5 = -0.005
                W += -0.005  # W减小0.005
            
            第2次更新 (梯度仍是0.5):
                v = 0.9 * (-0.005) - 0.01 * 0.5 = -0.0095
                W += -0.0095  # W减小0.0095（比上次多！）
            
            第3次更新 (梯度仍是0.5):
                v = 0.9 * (-0.0095) - 0.01 * 0.5 = -0.01355
                W += -0.01355  # W减小0.01355（越来越快！）
            
            → 这就是动量的加速效果！
        """
        # 对v进行初始化（只在第一次调用时执行）
        if self.v is None:
            self.v = {}
            # 为每个参数创建对应的速度向量，初始值为0
            # 使用zeros_like保证形状完全匹配
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 遍历所有参数，应用动量更新
        for key in params.keys():
            # 步骤1: 更新速度
            # v = 动量因子 * 旧速度 - 学习率 * 当前梯度
            # 负号是因为我们要往梯度的反方向走（梯度下降）
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]

            # 步骤2: 用速度更新参数
            # 参数沿着速度方向移动
            # 注意这里是 += 而不是 -=，因为v已经包含了负号
            params[key] += self.v[key]


# AdaGrad
class AdaGrad:
    """
    AdaGrad优化器
    AdaGrad优化器是一种自适应学习率的优化算法，
    它根据参数的梯度历史来调整学习率。

    """

    def __init__(self, lr=0.01):
        """
        初始化AdaGrad优化器

        参数:
            lr (float): 学习率，默认0.01
                       控制每次更新的步长大小

        属性:
            self.h (dict): 历史梯度平方字典，为每个参数维护一个历史梯度平方
                          初始为None，第一次update时初始化
                          形状与对应参数相同，初始值为0
        """
        self.lr = lr  # 学习率
        self.h = None  # 历史梯度平方

    # 更新参数
    def update(self, params, grads):

        # 对h进行初始化（只在第一次调用时执行）
        if self.h is None:
            self.h = {}
            # 为每个参数创建对应的速度向量，初始值为0
            # 使用zeros_like保证形状完全匹配
            for key, val in params.items():
                self.h[key] = np.zeros_like(val,dtype=np.float64)  # 确保浮点运算

        # 遍历所有参数，应用动量更新
        for key in params.keys():
            # 步骤1: 更新历史梯度平方
            # h = 历史梯度平方 + 当前梯度的平方
            self.h[key] += grads[key] ** 2
            # 步骤2: 用历史梯度平方更新参数
            # 添加一个微小量，防止梯度平方为0
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)


class RMSProp:
    def __init__(self, lr=0.01, decay_rate=0.9):
        self.lr = lr  # 学习率
        self.h = None  # 历史梯度平方
        self.decay_rate = decay_rate  # 衰减因子

    # 更新参数
    def update(self, params, grads):
        # 对h进行初始化（只在第一次调用时执行）
        if self.h is None:
            self.h = {}
            # 为每个参数创建对应的速度向量，初始值为0
            # 使用zeros_like保证形状完全匹配
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        # 遍历所有参数,按照公式进行更新
        for key in params.keys():
            # 步骤1: 更新历史梯度平方
            # h = 历史梯度平方 * 衰减因子 + 当前梯度的平方 * (1 - 衰减因子)
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] ** 2
            # 步骤2: 用历史梯度平方更新参数
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)


'''
Adam = Momentum + RMSProp + 偏差修正

优势:
✅ 自适应学习率（每个参数有自己的学习率）
✅ 动量加速（考虑历史梯度方向）
✅ 偏差修正（解决初始阶段估计不准）
✅ 几乎不需要调参（默认参数就很有效）

适用场景:
- 深度学习训练的首选优化器
- 特别适合稀疏梯度或非平稳目标
- 对超参数相对不敏感

'''
class Adam:
    """
    Adam (Adaptive Moment Estimation) 优化器
    
    结合了Momentum和RMSProp的优点:
        - 使用一阶矩估计(动量)来加速收敛
        - 使用二阶矩估计(自适应学习率)来调整每个参数的学习率
        - 通过偏差修正来解决初始阶段估计不准确的问题
    
    更新公式:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t      # 一阶矩(动量)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2    # 二阶矩(未中心化的方差)
        m_hat = m_t / (1 - beta1^t)                     # 偏差修正后的一阶矩
        v_hat = v_t / (1 - beta2^t)                     # 偏差修正后的二阶矩
        param -= lr * m_hat / (sqrt(v_hat) + eps)
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        """
        初始化Adam优化器
        
        参数:
            lr (float): 学习率，默认0.001（Adam通常使用较小的学习率）
            beta1 (float): 一阶矩估计的衰减率，默认0.9
                          控制动量的平滑程度
            beta2 (float): 二阶矩估计的衰减率，默认0.999
                          控制自适应学习率的平滑程度
        
        属性:
            self.iter (int): 迭代次数计数器，用于偏差修正
            self.v (dict): 一阶矩估计（动量），初始为None
            self.h (dict): 二阶矩估计（梯度平方），初始为None
            self.eps (float): 微小值，防止除零错误
        """
        self.lr = lr  # 学习率
        self.beta1 = beta1  # 一阶矩衰减率（动量因子）
        self.beta2 = beta2  # 二阶矩衰减率
        self.iter = 0  # 迭代次数
        self.m = None  # 一阶矩估计（动量）
        self.v = None  # 二阶矩估计（梯度平方的指数加权平均）
        self.eps = 1e-8  # 防止除零的微小值

    # 更新参数
    def update(self, params, grads):
        """
        使用Adam算法更新网络参数
        
        参数:
            params (dict): 参数字典，包含需要更新的权重和偏置
            grads (dict): 梯度字典，包含各参数的梯度
        
        更新流程:
            1. 首次调用时，初始化一阶矩m和二阶矩v（全零）
            2. 迭代次数加1
            3. 计算偏差修正后的学习率
            4. 对每个参数:
               a. 更新一阶矩估计 m
               b. 更新二阶矩估计 v
               c. 使用修正后的矩估计更新参数
        
        为什么要偏差修正?
            - 初始阶段 m 和 v 都接近0，会导致更新步长过小
            - 通过除以 (1 - beta^t) 来修正这种偏差
            - 随着 t 增大，修正因子趋近于1，影响逐渐减小
        """
        # 初始化一阶矩m和二阶矩v（只在第一次调用时执行）

        # m(moment) - 一阶矩估计（动量）
        # v(variance) - 二阶矩估计（梯度平方的指数加权平均）
        # 符合Adam算法的标准命名
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                # 确保初始化为float64类型，避免类型转换问题
                self.m[key] = np.zeros_like(val, dtype=np.float64)
                self.v[key] = np.zeros_like(val, dtype=np.float64)
        
        # 迭代次数加1
        self.iter += 1
        
        # 计算偏差修正后的学习率
        # 这个公式确保了初始阶段的更新不会太小
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        
        # 遍历所有参数，按照Adam公式进行更新
        for key in params.keys():
            # 确保grads[key]是float64类型，避免类型不匹配
            grad = np.asarray(grads[key], dtype=np.float64)
            
            # 步骤1: 更新一阶矩估计（动量）
            # m = beta1 * m_old + (1 - beta1) * grad
            # 简化形式: m += (1 - beta1) * (grad - m_old)
            self.m[key] += (1 - self.beta1) * (grad - self.m[key])
            
            # 步骤2: 更新二阶矩估计（梯度平方的指数加权平均）
            # v = beta2 * v_old + (1 - beta2) * grad^2
            # 简化形式: v += (1 - beta2) * (grad^2 - v_old)
            self.v[key] += (1 - self.beta2) * (grad ** 2 - self.v[key])
            
            # 步骤3: 使用修正后的矩估计更新参数
            # param -= lr_t * m / (sqrt(v) + eps)
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)
            