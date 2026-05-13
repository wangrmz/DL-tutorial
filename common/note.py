# 鲁棒性（Robustness）是中文对英文 "Robustness" 的音译 + 意译，核心意思是系统的“强壮性”或“抗干扰能力”。在机器学习和深度学习中，它通常指：
#
# 模型在面对输入数据有微小变化、噪声、错误、扰动甚至恶意攻击时，依然能保持稳定、正确的预测能力。
#
# 通俗理解
# 一个“鲁棒”的模型不会因为一张图片里加了一点肉眼看不见的噪声，就把“熊猫”错认成“长臂猿”。
#
# 就像一辆“鲁棒”的越野车，在泥泞、颠簸的路上也能正常行驶，不会轻易抛锚。

# 为什么重要？
# 真实世界的数据往往不完美（有噪声、缺失、脏数据）。一个鲁棒性差的模型，在实验室里满分，一上线就翻车。
#
# 安全性要求高的领域（自动驾驶、医疗诊断、金融风控）尤其需要鲁棒性，否则后果严重。
#
# 如何提升鲁棒性？
# 数据增强（训练时加噪声、旋转、裁剪等）
#
# 正则化（Dropout、权重衰减）
#
# 对抗训练（在训练中引入对抗样本）
#
# 集成方法（多个模型投票）
#
# 使用鲁棒的损失函数（如 Huber Loss 替代 MSE）
#
# 简单总结：鲁棒性 = 模型在“受干扰”的情况下依然靠谱，是衡量模型实用价值的关键指标之一。

# PyTorch ，TensorFlow

import  torch
# 检查 MPS 是否可用，如果输出 True 就说明 GPU 加速已经就绪
print(torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")  # 检查NVIDIA GPU，Mac电脑基本不会走这分支
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # 适用于M 系列芯片 Mac 电脑
else:
    device = torch.device("cpu")   # 兜底方案

print(f"当前使用的计算设备: {device}")









