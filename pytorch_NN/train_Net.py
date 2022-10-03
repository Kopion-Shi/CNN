import torch
from torch import nn
from d2l import torch as d2l
from pytorch_NN.backbones import model_Net
import time


def timer(func):
    def func_in():
        start_time = time.time()
        func()
        end_time = time.time()
        spend_time = (end_time - start_time)/60
        print("Spend_time:{} min".format(spend_time))
    return func_in

class TrainNet():
    def __init__(self,net):
        self.net_name=net
        self.net= model_Net.Net_CNN(net).run()
        self.run()

    def run(self):
        # conf.
        lr, num_epochs =0.01 , 10
        train_iter, test_iter=self.load_data()
        self.train_ch6(train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    #加载数据
    def load_data(self):
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=self.batch_size)
        return train_iter, test_iter

    # 由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中
    def evaluate_accuracy_gpu(self, data_iter, device=None):
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(self.net, nn.Module):
            self.net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(self.net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            # print(data_iter)
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

    # @save
    # @timer
    def train_ch6(self, train_iter, test_iter, num_epochs, lr, device):
        """用GPU训练模型(在第六章定义)"""

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_weights)
        print('training on', device)
        self.net.to(device)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'test acc'])
        timer, num_batches = d2l.Timer(), len(train_iter)
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = d2l.Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
            test_acc = self.evaluate_accuracy_gpu(test_iter)
            print('epoch:',epoch+1,'train_l',train_l,'train_acc',train_acc,'test_acc',test_acc)
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')