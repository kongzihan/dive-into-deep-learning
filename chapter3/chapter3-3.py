"""
本程序包含了《动手学习深度学习》中，第三章 深度学习基础中的部分代码
代码为pytorch版本

线性回归的简洁实现 -- 用 pytorch
"""
import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

# ======================    生成数据集     ====================
num_inuts = 2
num_examples = 1000

# 真实的 weight 和 bias
true_w = [2, -3.4]
true_b = 4.2

# features 是训练数据特征 x， labels 是标签 y
features = torch.tensor(
    np.random.normal(0, 1, (num_examples, num_inuts)),
    dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

# ======================    读取数据集     ====================
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


# 读取并打印第一个小批量数据样本
# for X, y in data_iter:
#     print(X, y)
#     break


# ======================    定义模型     ====================
# 首先，导入 torch.nn 模块，nn为神经网络 neural network 的缩写
# 这个模块定义了大量神经网络的层
# nn 的核心数据结构是 module， 这是一个抽象概念，既可以表示神经网络中的某个层 layer，
# 也可以表示一个包含很多层的神经网络
# 实际使用中，最常见的做法是继承 nn.module, 撰写自己的网络/层
# 一个 nn.module 实例应该包含一些层以及返回输出的前向传播方法
# 下面用 nn.Module 实现一个线性回归模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inuts)
# 打印网络结构
# print(net)

# 也可以用 nn.Sequential 来更方便地搭建网络
# Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中
# 当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inuts, 1))
# print(net)

# 可以通过 net.parameters() 来查看模型所有的可学习参数，此函数将返回一个生成器
for param in net.parameters():
    print(param)

# tips
# 单样本转换为 min_batch
# torch.nn 仅支持输入一个batch的样本不支持单个样本输入
# 如果只有单个样本，可使用 input.unsqueeze(0)来添加一维
# 0表示在张量最外层加一个中括号变成第一维。


# ======================    初始化模型参数     ====================
# net 为自定义linearNet
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# net 为 moduleList 或者 sequential 实例时才可以
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)


# ======================    定义损失函数     ====================
# 损失函数可以看作是一种特殊的层
# pytorch 将这些损失函数实现为 nn.module 的子类
# 使用它提供的均方误差损失作为模型的损失函数
loss = nn.MSELoss()

# ======================    定义优化算法     ====================
optimiter = optim.SGD(net.parameters(), lr=0.03)
print(optimiter)

# ======================    训练模型     ====================
# 调用optim实例的step函数来迭代模型参数
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        # 前向传播
        output = net(X)
        # 计算损失
        l = loss(output, y.view(-1, 1))  # view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
        # 梯度清零
        optimiter.zero_grad()
        # 反向传播
        l.backward()
        # 更新参数
        optimiter.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 比较学到的模型参数和真实的模型参数
# 从net处获得需要的层，并访问其权重 weight 和 偏差 bias
dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)
