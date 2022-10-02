import torch
import d2l
import model_Net



net=model_Net.Net_CNN('LeNet').run()
# net= model_Net.Net_CNN('LeNet').net_LeNet()
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)