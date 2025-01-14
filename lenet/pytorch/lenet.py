import torch
from torch import nn
from torch.nn import functional as F

import os
import struct

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('input: ', x.shape)
        x = F.relu(self.conv1(x))
        print('conv1',x.shape)
        x = self.pool1(x)
        print('pool1: ', x.shape)
        x = F.relu(self.conv2(x))
        print('conv2',x.shape)
        x = self.pool1(x)
        print('pool2',x.shape)
        x = x.view(x.size(0), -1)
        print('view: ', x.shape)
        x = F.relu(self.fc1(x))
        print('fc1: ', x.shape)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

 
def model_onnx():
    input = torch.ones(1, 1, 32, 32, dtype=torch.float32).cuda()
    model = Lenet5()
    model = model.cuda()
    torch.onnx.export(model, input, "./lenet.onnx", verbose=True)

#将模型权重按照key,value形式存储为16进制文件 
def Inf():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('lenet5.pth')
    net = net.to('cuda:0')
    net.eval()
    #print('model: ', net)
    #print('state dict: ', net.state_dict()['conv1.weight'])
    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    #print('input: ', tmp)
    out = net(tmp)
    print('lenet out:', out)

    f = open("lenet5.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        #print('key: ', k)
        #print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


def main():
    print('cuda device count: ', torch.cuda.device_count())
    torch.manual_seed(1234)
    net = Lenet5()
    net = net.to('cuda:0')
    net.eval()
    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    out = net(tmp)
    print('lenet out shape:', out.shape)
    print('lenet out:', out)
    torch.save(net, "lenet5.pth")

if __name__ == '__main__':
    #main()
    #model_onnx()
    Inf()