import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class AlexNet(nn.Module):
    """
    The input for Alexnet is a 224*224 RGB image
    Args:
         num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self):
        super(AlexNet,self).__init__()
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        self.conv1=nn.Conv2d(1,96, kernel_size=11, stride=4, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=3, stride=2)
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        self.conv2=nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        self.conv3=nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4=nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5=nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.flatten=nn.Flatten()
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        self.linear1=nn.Linear(6400, 4096)
        self.dropout=nn.Dropout(p=0.5)
        self.linear2=nn.Linear(4096, 4096)
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        self.linear3=nn.Linear(4096, 10)
        self.relu=nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x=self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        y=self.linear3(x)
        return y

if __name__=='__main__':
    net = AlexNet()
    writer = SummaryWriter('./data/tensorboard')
    writer.add_graph(net, input_to_model=torch.rand(1, 1, 224, 224))
    writer.close()

    