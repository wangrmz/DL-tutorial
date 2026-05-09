import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 读取数据
def get_data():
    """
    加载并预处理MNIST手写数字数据集
    
    返回:
        tuple: (X_test, X_train, y_test, y_train)
            - X_test: 测试集特征，形状 (N_test, 784)，已归一化到[0,1]
            - X_train: 训练集特征，形状 (N_train, 784)，已归一化到[0,1]
            - y_test: 测试集标签，形状 (N_test,)，值为0-9的整数
            - y_train: 训练集标签，形状 (N_train,)，值为0-9的整数
    
    数据处理流程:
        1. 从CSV文件读取原始数据
        2. 分离特征和标签
        3. 按8:2比例划分训练集和测试集
        4. 对特征进行归一化处理（缩放到[0,1]范围）
        5. 转换为numpy数组格式
    
    注意:
        - 使用random_state=42保证每次运行结果可复现
        - 归一化时使用fit_transform处理训练集，只用transform处理测试集
          这是为了防止数据泄露（data leakage）
        - MNIST图像是28x28像素，展平后为784维向量
    """
    # 1.读取数据集
    # 从CSV文件加载数据，包含784个像素特征和1个标签列
    data = pd.read_csv('../data/train.csv')
    
    # 2.划分数据集
    # 分离特征和标签
    X = data.drop('label', axis=1)  # 删除标签列，保留所有像素特征（784列）
    y = data['label']               # 提取标签列（0-9的数字）
    
    # 按80%训练集、20%测试集的比例随机划分
    # random_state=42确保每次运行得到相同的划分结果（可复现性）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 测试集占比20%
        random_state=42     # 随机种子，保证结果可复现
    )
    
    # 3.特征工程：归一化处理
    # 创建MinMaxScaler对象，将特征值缩放到[0, 1]范围
    # 公式: X_scaled = (X - X_min) / (X_max - X_min)
    scaler = MinMaxScaler()
    
    # 在训练集上fit_transform：计算最大值最小值，并进行转换
    # 这样scaler就"学习"了训练集的统计信息
    X_train = scaler.fit_transform(X_train)
    
    # 在测试集上只transform：使用训练集的统计信息进行转换
    # 重要：不能用测试集fit，否则会造成数据泄露！
    X_test = scaler.transform(X_test)
    
    # 4.将数据转成ndarray
    # pandas Series转换为numpy数组，便于后续神经网络处理
    y_train = y_train.values
    y_test = y_test.values
    
    # 返回顺序：先测试集后训练集（注意！）
    return X_test, X_train, y_test, y_train
