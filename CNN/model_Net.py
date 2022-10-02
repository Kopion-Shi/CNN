import torch
from torch import nn
from d2l import torch as d2l

class Net_CNN():
    def __init__(self,net_name,):
        self.net_name=net_name

    def run(self):
        net_dict={
            'LeNet':self.net_LeNet,
        }
        net=net_dict[self.net_name]()
        return net

    # @property
    def net_LeNet(self):
        return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))


    