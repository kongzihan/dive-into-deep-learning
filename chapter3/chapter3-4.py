"""
本程序包含了《动手学习深度学习》中，第三章 深度学习基础中的部分代码
代码为pytorch版本

图像分类数据集
 fashion MNIST 是一个10类服饰分类数据集
 -- 用 pytorch
"""
import torchvision
import torchvision.transforms as transforms
import d2lzh as d2l
from time import time


# ======================    获取数据集     ====================
# 通过 torchvision 的torchvision.datasets 来下载这个数据集
# 指定参数 transform=transforms.ToTensor() 使所有数据转换为 Tensor
# 将尺寸为 H x W x C 且数据位于[0,255]的PIL图片或者数据类型为np.uint8的numpy
# 数组转换为尺寸为 C x H x W且数据类型为 torch.float32 且位于[0.0,1]的tensor
mnist_train = torchvision.datasets.FashionMNIST(
    root='data/FashionMNIST', train=True, download=True,
    transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(
    root='data/FashionMNIST', train=False, download=True,
    transform=transforms.ToTensor())

# 训练集和测试集中的每个类别的图像数分别为6,000和1,000
# 有10个类别
# print(len(mnist_train))
# print(len(mnist_test))

# 通过方括号访问样本
# 获取第一个样本的图像和标签
# feature, label = mnist_train[0]
# print(feature.shape)
# print(feature.dtype)
# print(label)
# print(type(label))


# 下面定义一个可以在一行里画出多张图像的对应标签的函数
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    # 这里的_表示我们忽略（不适用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    d2l.plt.show()


# 看一下训练数据集中前10个样本的图像内容和文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirts', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


# 使用多进程加速数据读取
# pytorch 中的 DataLoader允许使用多进程来加速数据读取
batch_size = 256
# if sys.platform.startswith('win'):
#     # 0表示不用额外的进程来加速读取数据
#     num_workers = 0
# else:
#     num_workers = 4
# train_iter = torch.utils.data.DataLoader(
#     mnist_train, batch_size=batch_size, shuffle=True,
#     num_workers=num_workers)
# test_iter = torch.utils.data.DataLoader(
#     mnist_test, batch_size=batch_size, shuffle=False,
#     num_workers=num_workers)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 查看读取一遍训练数据需要的时间
start = time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time()-start))








