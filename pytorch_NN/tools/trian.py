# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 11:59
# @Author  : 石鑫磊
# @Site    : 
# @File    : trian.py
# @Software: PyCharm 
# @Comment :
from pytorch_NN.data_preprocessing.loaddate import Date
import torch
from torch import nn


# class Animator:
#     """For plotting data in animation."""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         """Defined in :numref:`sec_softmax_scratch`"""
#         # Incrementally plot multiple lines
#         if legend is None:
#             legend = []
#         self.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         # Use a lambda function to capture arguments
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts
#
#     def use_svg_display(self):
#         """Use the svg format to display a plot in Jupyter.
#
#         Defined in :numref:`sec_calculus`"""
#         backend_inline.set_matplotlib_formats('svg')
#
#
#     def add(self, x, y):
#         # Add multiple data points into the figure
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

import numpy as np
import time


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


from pytorch_NN.config.backbones .AlexNet import AlexNet
class Train():
    def __init__(self):
        get_dataloader_workers = 8
        batch_size = 128
        resize = 224
        self.train_iter= Date(get_dataloader_workers,batch_size).train_iter(resize)
        self.test_iter = Date(get_dataloader_workers,batch_size).test_iter(resize)
        self.net= AlexNet().feature()

    def accuracy(self,y_hat, y):
        """Compute the number of correct predictions.
        Defined in :numref:`sec_softmax_scratch`"""
        argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
        reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
        astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))

    # 由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中
    def evaluate_accuracy_gpu(self, data_iter, device=None):
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(self.net, nn.Module):
            self.net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(self.net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = Accumulator(2)
        with torch.no_grad():
            # print(data_iter)
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(self.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

    def try_gpu(self,i=0):
        """Return gpu(i) if exists, otherwise return cpu().

        Defined in :numref:`sec_use_gpu`"""
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    def train_ch6(self, num_epochs, lr):
        """Train a model with a GPU (defined in Chapter 6).

        Defined in :numref:`sec_lenet`"""

        device=self.try_gpu()

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_weights)
        # print('training on', device)
        self.net.to(device)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
        #                         legend=['train loss', 'train acc', 'test acc'])  #画图
        timer, num_batches = Timer(), len(self.train_iter)
        # print(num_epochs)
        for epoch in range(num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(self.train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0],self.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
            test_acc = self.evaluate_accuracy_gpu(self.test_iter)
            print('epoch：num{}'.format(epoch),f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}',
                  f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}',
                  'run time:{}'.format(timer.sum()))
            print('=========================================================')
        print('train over',
              'run time:{}'.format(timer.sum(),
            f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}',
              f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}'))


if __name__=='__main__':
    net=Train()
    lr, num_epochs = 0.01, 10
    net.train_ch6(lr=lr, num_epochs=num_epochs)
