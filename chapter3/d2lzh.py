from IPython import display
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch


# 在作图前只需要调用 d2lzh.set_figsize() 即可打印矢量图并设置图的尺寸。
def use_svg_display():
    # 用矢量图显示
    # set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


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