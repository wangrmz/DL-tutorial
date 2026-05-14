import torch

'''
Dropout（随机失活，暂退法）是一种在学习的过程中随机关闭神经元的方法。
可以通过torch.nn.Dropout(p)来使用Dropout，并通过参数p来设置失活概率。

'''

dropout = torch.nn.Dropout(p=0.5)
x = torch.randint(1, 10, (10,), dtype=torch.float32)
print("Dropout前：", x)
print("Dropout后：", dropout(x))

'''
Dropout前： tensor([7., 1., 3., 2., 8., 2., 8., 2., 9., 7.])
Dropout后： tensor([ 0.,  0.,  0.,  0., 16.,  0., 16.,  0.,  0.,  0.])
'''