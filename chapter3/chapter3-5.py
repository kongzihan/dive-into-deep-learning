"""
本程序包含了《动手学习深度学习》中，第三章 深度学习基础中的部分代码
代码为pytorch版本

softmax 回归的从0开始实现
 fashion MNIST 是一个10类服饰分类数据集
 -- 用 pytorch
"""
import torch
import numpy as np
import d2lzh as d2l


# =======================  实现 softmax 运算  ======================
# dim=0 同列元素求和
# dim=1 同行元素求和
# 定义softmax运算
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    # 这里应用了广播机制
    return X_exp / partition


# =======================  定义模型  ======================
# 通过view函数将每张原始图像改成长度为 num_inputs 的向量
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), w) + b)


# =======================  定义损失函数  ======================
def cross_entropy(y_hat, y):
    # 使用gather函数，得到2个样本的标签的预测概率
    # torch.gather(input=y_hat, dim=1, index=y.view(-1, 1))
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


# =======================  计算分类准确率  ======================
def accuracy(y_hat, y):
    # y_hat_.argmax(dim=1) （把列数变成1）返回矩阵 y_hat 每行中最大元素的索引
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 评价模型net在数据集 data_iter 上的准确率
def evaluate_accuracy(data_iter):
    acc_sum, n = 0.0, 0
    # 特征和标签
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 样本个数
        n += y.shape[0]
    return acc_sum / n


# =======================  训练模型  ======================
def train_ch3(train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()




# =======================  读取数据集  ======================
# 使用 Fashion-MNIST 数据集，并设置批量大小为 256
batch_size_ = 256
train_iter_, test_iter_ = d2l.load_data_fashion_mnist(batch_size_)

# =======================  初始化模型参数  ======================
# 使用向量表示每个样本
# 每个样本输入是高和宽均为28像素的图像
# 模型的输入向量的长度是 28x28=784，该向量的每个元素对应图像中每个像素
# 由于图像有10个类别，单层神经网络输出层的输出个数为10
# 因此，softmax回归的权重和偏差参数分别为 784x10 和 1x10 的矩阵
num_inputs = 784
num_outputs = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)),
                 dtype=torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)
# 需要模型参数梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 设置 x和y
X_ = torch.tensor(np.random.normal(0, 1, (2, 5)), dtype=torch.float32)
X_prob = softmax(X_)
# y_hat 是2个样本在3个类别的预测概率
y_hat_ = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 变量y是这2个样本的标签类别
y_ = torch.LongTensor([0, 2])

# 设置训练参数
num_epochs_, lr_ = 5, 0.1

# 随机初始化模型后，输出模型的准确率（因为有10个类别，所以预测结果应接近于0.1）
print(evaluate_accuracy(test_iter_))
# 结果为 0.0563

# 训练模型
# 打印训练过程中的预测结果



