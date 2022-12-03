"""
本程序包含了《动手学习深度学习》中，第三章 深度学习基础中的部分代码
代码为pytorch版本

线性回归从0开始实现
"""
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from IPython import display

# ======================    生成数据集     ====================
# 输入个数为2
num_inputs = 2
# 样本数为1000
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

# features 的每一行是一个长度为2的向量
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# 而 labels 的每一行是一个长度为1的向量，y = xw + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# y = xw + b + e, 其中噪声项e服从正态分布 N(0, 0.01)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

print(features[0])
print(labels[0])


# 生成第二个特征features[:,1]和标签labels的散点图，
# 可以更直观地观察两者间的线性关系
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()


# ===========================     读取数据集      ==========================
# 训练模型时，需要遍历数据集并不断读取小批量数据样本
# 每次返回 batch_size 个随机样本的特征和标签
def data_iter(batch_size_, features_, labels_):
    num_examples_ = len(features_)
    indices = list(range(num_examples_))
    # 样本的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples_, batch_size_):  # range的三个参数，start，stop，step
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i:min(i+batch_size_,
                                           num_examples_)])
        # index_select函数根据索引返回对应元素
        yield features_.index_select(0, j), labels_.index_select(0, j)


# 读取第一个小批量数据样本并打印
# 每个批量的特征形状为（10,2），分别对应批量大小和输入个数；
# 标签形状为批量大小
batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

# =======================  初始化模型参数  ======================
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),
                 dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 在后面的模型训练中，需要对这些参数求梯度来迭代参数的值
# 因此需要让它们的 requires_grad=true
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# =======================  定义模型  ======================
# 下面是线性回归的矢量计算表达式的实现
# 使用mm()函数做矩阵乘法
def linreg(X_, w_, b_):
    return torch.mm(X_, w_) + b_


# =======================  定义损失函数  ======================
# 使用平方损失来定义线性回归的损失函数
# 需要把真实值预测成 y_hat(y^) 的形状
def squared_loss(y_hat, y_):
    return (y_hat - y_.view(y_hat.size())) ** 2 / 2


# =======================  定义优化算法  ======================
# 以下的 sgd 函数实现了小批量随机梯度下降算法，
# 不断迭代模型参数来优化损失函数
# 这里自动求梯度模块计算的得来的梯度是一个批量样本的梯度和
# 将它除以批量大小来得到平均值
def sgd(params, lr_, batch_size_):
    for param in params:
        param.data -= lr_ * param.grad / batch_size_


# =======================  训练模型  ======================
# 一个epoch就是一个迭代周期
lr = 0.3
num_epochs = 3
net = linreg
loss = squared_loss
# 训练模型一共需要 num_epochs 个迭代周期
for epoch in range(num_epochs):
    # 在一个 epoch 中，会使用训练集中所有样本一次
    # x和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        # l 是有关小批量 x和y 的损失
        # sum 求和，得到标量
        l_ = loss(net(X, w, b), y).sum()
        # 小批量损失对模型参数求梯度
        l_.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w, b], lr, batch_size)
        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))










