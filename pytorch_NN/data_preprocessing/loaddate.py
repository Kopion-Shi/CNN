# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 5:16
# @Author  : 石鑫磊
# @Site    : 
# @File    : loaddate.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

class Date():
    def __init__(self,get_dataloader_workers,batch_size):
        self.get_dataloader_workers=get_dataloader_workers
        self.batch_size=batch_size

    def load_data_fashion_mnist(self, resize=None):
        """Download the Fashion-MNIST dataset and then load it into memory.
        Defined in :numref:`sec_fashion_mnist`"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True)
        (train_iter,test_iter) = (data.DataLoader(mnist_train, self.batch_size, shuffle=True,
                                num_workers=self.get_dataloader_workers),
                data.DataLoader(mnist_test,self.batch_size, shuffle=False,
                                num_workers=self.get_dataloader_workers))
        return train_iter,test_iter

    def train_iter(self,resize):
        return self.load_data_fashion_mnist(resize)[0]
    def test_iter(self,resize):
        return self.load_data_fashion_mnist(resize)[1]
if __name__=='__main__':

    data_load=Date(get_dataloader_workers=8,batch_size=128)
    train_iter, test_iter = data_load.train_iter(resize=224),data_load.test_iter(resize=224)