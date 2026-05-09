import numpy as np

# 数值微分求导，传入x是一个标量
def numerical_diff(f, x):
    h = 1e-4  # 微小值
    return (f(x + h) - f(x - h)) / (2 * h) # 中心差分


# 使用数值微分求梯度，传入x是一个向量
def _numerical_gradient(f, x):
    h = 1e-4  # 微小值
    grad = np.zeros_like(x)
    # 遍历x中的特征xi
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 利用中心差分公式计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 恢复x[idx]的值
        x[idx] = tmp_val
    return grad

# 传入X是一个矩阵
def numerical_gradient(f, X):
    # 一维
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        # 二维，对每一行进行数据，分别求梯度
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient(f, x)
        return grad

'''
enumerate(X) 会生成一系列 (索引, 元素) 的元组。在 for idx, x in enumerate(X): 
这个循环中：
idx 依次得到 0, 1, 2, ...（即当前元素在 X 中的位置索引）
x 依次得到 X[0], X[1], X[2], ...（即该索引位置上的元素值）

在 numerical_gradient 这段代码中，正是利用这个特性：对 X 的每一行（一个样本）调用 _numerical_gradient 计算该样本的梯度，
然后把结果放到 grad[idx] 对应行的位置，最终得到一个与 X 形状相同的梯度矩阵。
'''

X = np.array([[1, 2], [3, 4], [5, 6]])
for idx, x in enumerate(X):
    print(f"索引 {idx}: 值 {x}")
