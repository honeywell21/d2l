import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys

sys.path.append("..")

import torchvision
from IPython import display
from numpy import argmax
import torchvision.transforms as transforms
from time import time
import matplotlib.pyplot as plt
import numpy as np

batch_size = 256

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

# 生成迭代器
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        # 定义一个输入层

    # 定义向前传播（在这个两层网络中，它也是输出层）
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        # 将x换形为y后，再继续向前传播
        return y


net = LinearNet(num_inputs, num_outputs)

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 5


# 一共进行五个学习周期

# 计算准确率
def net_accurary(data_iter, net):
    right_sum, n = 0.0, 0
    for X, y in data_iter:
        # 从迭代器data_iter中获取X和y
        right_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 计算准确判断的数量
        n += y.shape[0]
        # 通过shape[0]获取y的零维度（列）的元素数量
    return right_sum / n


def get_Fashion_MNIST_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
    # labels是一个列表，所以有了for循环获取这个列表对应的文本列表


def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')
    # 绘制矢量图
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # 设置添加子图的数量、大小
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer, net_accurary):
    for epoch in range(num_epochs):
        # 损失值、正确数量、总数 初始化。
        train_l_sum, train_right_sum, n = 0.0, 0.0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 数据集损失函数的值=每个样本的损失函数值的和。
            optimizer.zero_grad()  # 对优化函数梯度清零
            l.backward()  # 对损失函数求梯度
            optimizer.step()

            train_l_sum += l.item()
            train_right_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = net_accurary(test_iter, net)  # 测试集的准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_right_sum / n, test_acc))


train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer, net_accurary)

X, y = iter(test_iter).next()

true_labels = get_Fashion_MNIST_labels(y.numpy())
pred_labels = get_Fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
