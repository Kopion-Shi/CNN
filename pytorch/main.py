# -*- coding: utf-8 -*-
# @Time    : 2022/9/17 9:46
# @Author  : 石鑫磊
# @Site    : 
# @File    : main.py
# @Software: PyCharm 
# @Comment :
import train_Net
class Main():
    def __init__(self,net_name,batch_size):
        self.net=net_name
        self.batch_size=batch_size
        self.run()

    def run(self):
        train_Net.TrainNet(self.net,self.batch_size)


if __name__=='__main__':
    net=Main('LeNet',batch_size=256)
