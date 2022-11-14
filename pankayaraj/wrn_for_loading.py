from ast import arg
from lib2to3.pgen2.grammar import opmap_raw
from operator import mod
from os import pread
from turtle import pen, st
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn
from argparse import ArgumentParser
import torch.nn.functional as F
import utils
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_layers_nextwork_block(layer_list, inx, n):
    j = inx
    layers_network = []
    for i in range(n):
        if i == 0:
            layers_network.append(layer_list[j:j+7])
            j = j+7
        else:
            layers_network.append(layer_list[j:j+6])
            j = j+6
    return layers_network


def func_w_index_check( module_no, start_index, end_index, x, *funs):
    out = x
    if start_index <= module_no and module_no < end_index:
        
        for f in funs:
            out = f(out)
            
    else:
        pass
    
    return out


class BasicBlock_Loading(nn.Module):
    
    def __init__(self, layer_list, in_planes, out_planes, stride,  dropRate=0.0):
        super(BasicBlock_Loading, self).__init__()
        self.bn1 = layer_list[0]
        self.relu1 = layer_list[1]
        self.conv1 = layer_list[2]
        self.bn2 = layer_list[3]
        self.relu2 = layer_list[4]
        self.conv2 = layer_list[5]
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and layer_list[6] or None

    def forward(self, x, module_no, start_indx, end_indx):
        if not self.equalInOut:
            x = func_w_index_check(module_no, start_indx, end_indx, x, self.bn1, self.relu1)
        else:
            out = func_w_index_check(module_no, start_indx, end_indx, x, self.bn1, self.relu1)
        
        module_no += 2
        
        if self.equalInOut:
            out = func_w_index_check(module_no, start_indx, end_indx, out, self.conv1, self.bn2, self.relu2)
        else:
            out = func_w_index_check(module_no, start_indx, end_indx, x, self.conv1, self.bn2, self.relu2)
            
        
        module_no += 3

        if self.droprate > 0:
            if start_indx <= module_no and module_no < end_indx:
                out = F.dropout(out, p=self.droprate, training=self.training)
            else:
                pass
        
        out = func_w_index_check(module_no, start_indx, end_indx, out, self.conv2)
        module_no += 1
        
        if not self.equalInOut:
        
            shortcut = func_w_index_check(module_no, start_indx, end_indx, x, self.convShortcut)
            module_no += 1
            if start_indx <= module_no and module_no < end_indx:
                return torch.add(shortcut, out), module_no, start_indx, end_indx
            else:
                return out, module_no, start_indx, end_indx

            
        else:
            if start_indx <= module_no and module_no < end_indx:
                return torch.add(x, out), module_no, start_indx, end_indx
            else:
                return out, module_no, start_indx, end_indx

class NetworkBlock_Loading(nn.Module):
    def __init__(self, layer_list, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        #here the layer_list is a list of lists with each element of length = 7 for each of the lower blocks if euqalInpOut
        #else 6
        super(NetworkBlock_Loading, self).__init__()
        self.layers = self._make_layer(layer_list, block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, layer_list, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        
        for i in range(nb_layers):
            layers.append(block(layer_list[i], i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.ModuleList(layers)

    def forward(self, x, module_no, start_indx, end_indx):
        inp = x
        for indx, l in enumerate(self.layers):
            inp, module_no, start_indx, end_indx = l(inp, module_no, start_indx, end_indx)
        return inp, module_no, start_indx, end_indx

class WideResNet_Loading(nn.Module):

    """
    PARAMETERS

    layer_list  :   the list of modules you get from the flattenTheNN function
    depth       :   for their given models it seems like 40
    num_classes :   10
    widen_factor:   seems like 2
    dropRate    :   0.0
    """


    def __init__(self, layer_list, depth, num_classes, widen_factor=2, dropRate=0.0):
        super(WideResNet_Loading, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock_Loading
        # 1st conv before any network block

        indx = 0
        self.conv1 = layer_list[indx]
        indx += 1

        # 1st block
        layers_for_module = calculate_layers_nextwork_block(layer_list, indx, n)
        self.block1 = NetworkBlock_Loading(layers_for_module, n, nChannels[0], nChannels[1], block, 1, dropRate)
        indx += 37

        # 2nd block
        layers_for_module = calculate_layers_nextwork_block(layer_list, indx, n)
        self.block2 = NetworkBlock_Loading(layers_for_module, n, nChannels[1], nChannels[2], block, 2, dropRate)
        indx += 37
        
        # 3rd block
        layers_for_module = calculate_layers_nextwork_block(layer_list, indx, n)
        self.block3 = NetworkBlock_Loading(layers_for_module, n, nChannels[2], nChannels[3], block, 2, dropRate)
        indx += 37

        # global average pooling and classifier
        self.bn1 = layer_list[indx]
        indx += 1 

        self.relu = layer_list[indx]
        indx += 1 

        self.fc = layer_list[indx]
        indx += 1 
        
        
        self.nChannels = nChannels[3]

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                
                m.bias.data.zero_()
        
    def forward(self, x, start_index, end_index):
        """
        We are supposed to break the network into two parts
        For the first part start_index = 0 and end_index = where ReLU is supposed to be. eg 2,5,9, 12 in the breakstart list
        For the second part of the network. Start_index = where ReLU is supposed to be. End Index = len(layer_list)
        """



        module_no = 0
        print(module_no)
        
        if start_index <= module_no and module_no < end_index:
            out = self.conv1(x)
        else:
            out = x
        module_no += 1
        
        out, module_no, start_index, end_index = self.block1(out, module_no, start_index, end_index)
        out, module_no, start_index, end_index = self.block2(out, module_no, start_index, end_index)
        out, module_no, start_index, end_index = self.block3(out, module_no, start_index, end_index)

        
        
        if module_no >= start_index and module_no < end_index:
            out = self.relu(self.bn1(out))
            module_no += 2
        else:
            pass

        
        if module_no >= start_index and module_no < end_index:
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            module_no += 1
            return self.fc(out)
        else:
            return out
        

"""


def flattenTheNN(model):
    #print("Calling get all children")
    ans = list(model.children())
    #print("DEBUG: (13)\t ans: ", ans)
    idx = 0

    while idx < len(ans):
        # print("----->oneIter",idx,ans)
        if len(list(ans[idx].children())) > 0:
            ans = ans[:idx] + list(ans[idx].children()) + ans[idx + 1:]
            # print("----->oneIter(changed)",idx,ans)
        else:
            idx += 1

    return ans


transformToNormalizedTensor = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

from wrn import WideResNet
import time
if __name__ == '__main__':

    batchSize = 5
    depth=40
    n = (depth - 4) // 6
    num_classes=10

    trainset = torchvision.datasets.CIFAR10(root='data', train=True,download=True, transform=transformToNormalizedTensor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='data', train=False,download=True, transform=transformToNormalizedTensor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=1)

    model_path = "../datasets/detection/train/trojan/id-0001/model.pt"
    model = torch.load(model_path)
    layer_list = flattenTheNN(model)
    break_points = []
    for lIdx in range(len(layer_list)):
        if isinstance(layer_list[lIdx],nn.ReLU):
            break_points.append([lIdx,lIdx+1])

    print("AA", len(break_points))
    L = list(model.children())
    for l in L:
        #print(L)
        pass
    #layer_list = torch.load("layer_list",map_location=torch.device('cpu'))
    #break_points = torch.load("index_list",map_location=torch.device('cpu'))
    import timeit


    T = 0
    # Your statements here


    print(break_points)
    m = 0
    for x, yTrue in testloader:
        start = timeit.default_timer()
        if m == 36:#len(break_points)
            break

        
        a = layer_list[0](x)

        widen_factor = 2
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        in_planes, out_planes, stride = nChannels[0], nChannels[1], 1
        
        #B1 = BasicBlock_new(layer_list[1:8], m == 0 and in_planes or out_planes, out_planes, m == 0 and stride or 1,)
        #B2 = BasicBlock( i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,)

        layers_network = []
        
        j = 1
        for i in range(n):
            if i == 0:
                layers_network.append(layer_list[j:j+7])
                j = j+7
            else:
                layers_network.append(layer_list[j:j+6])
                j = j+6
        
        
        

        #B2 = NetworkBlock_new(layers_network, n, nChannels[0], nChannels[1], BasicBlock_new, 1, 0.0)
       
        B = WideResNet_Loading(layer_list, depth, 10, widen_factor)
        C = WideResNet(depth,10,widen_factor)
        

        q = 10
        a = B(x, 0, break_points[q][0])
        b = B(a, break_points[q][0], len(layer_list))

        c = model(x)

        S = a.size()
        t = S[0]+S[1]+S[2]+S[3]

        #print(b,c)
        B.modules_no = 0
        m += 1

        stop = timeit.default_timer()

        T += (stop - start)*t
        print(T)
"""
