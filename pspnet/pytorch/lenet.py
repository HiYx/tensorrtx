import torch
from torch import nn
from torch.nn import functional as F

import os
import struct

 
def model_onnx():
    input = torch.ones(1, 1, 32, 32, dtype=torch.float32).cuda()
    model = Lenet5()
    model = model.cuda()
    torch.onnx.export(model, input, "./lenet.onnx", verbose=True)

#将模型权重按照key,value形式存储为16进制文件 
def Inf():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('./lenet5.pth')
    net = net.to('cuda:0')
    net.eval()
    #print('model: ', net)
    #print('state dict: ', net.state_dict()['conv1.weight'])
    tmp = torch.ones(1, 3, 256, 256).to('cuda:0')
    #print('input: ', tmp)
    # out = net(tmp)
    # print('lenet out:', out)



    f = open("lenet5.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        # vr = v.reshape(-1).cpu().numpy()
        # f.write("{} {}".format(k, len(vr)))
        # for vv in vr:
            # f.write(" ")
            # f.write(struct.pack(">f", float(vv)).hex())
        # f.write("\n")



if __name__ == '__main__':
    #model_onnx()
    Inf()