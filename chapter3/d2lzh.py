import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import sys


# 在作图前只需要调用 d2lzh.set_figsize() 即可打印矢量图并设置图的尺寸。
def use_svg_display():
    # 用矢量图显示
    # set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 读取数据集
def load_data_fashion_mnist(batch_size):
    # 获取数据集
    # 通过 torchvision 的torchvision.datasets 来下载这个数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='data/FashionMNIST', train=True, download=True,
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root='data/FashionMNIST', train=False, download=True,
        transform=transforms.ToTensor())
    # 使用多进程加速数据读取
    if sys.platform.startswith('win'):
        # 0表示不用额外的进程来加速读取数据
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter


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
def sgd(params, lr, batch_size_):
    for param in params:
        param.data -= lr * param.grad / batch_size_