# 输出层主要考虑softmax函数
import torch


x = torch.randn(3, 5)
print(x)

y = torch.softmax(x, dim=1)
print(y)