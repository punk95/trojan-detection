from ast import arg
import imp
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn
from argparse import ArgumentParser
import torch.nn.functional as F
import math


def func_w_index_check(skip_check, module_no, start_index, end_index, x, *funs):

    out = x
    if start_index <= module_no and end_index > module_no :
        for f in funs:
            out = f(out)

        skip_check[0] = True
    else:

        pass
    return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, ):
        super(BasicBlock, self).__init__()
        self.skip_check = False
        self.module_no = 0
        self.out_planes = out_planes
        self.bn1 = nn.BatchNorm2d(in_planes).to("cuda")
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False).to("cuda")

        self.bn2 = nn.BatchNorm2d(out_planes).to("cuda")
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False).to("cuda")
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False).to("cuda") or None

    def forward(self, x, module_no, start_index, end_index, skip_check):
        # func_w_index_check(self.skip_check, self.module_no, start_index)
        self.module_no = module_no

        if not self.equalInOut:
            x = func_w_index_check(skip_check, self.module_no, start_index, end_index, x, self.relu1, self.bn1)
        else:
            out = func_w_index_check(skip_check, self.module_no, start_index, end_index, x, self.relu1, self.bn1)
        self.module_no += 2

        # in this part there maybe a problem
        if self.equalInOut:

            out = func_w_index_check(skip_check, self.module_no, start_index, end_index, out, self.relu2, self.bn2, self.conv1)
        else:


            out = func_w_index_check(skip_check, self.module_no, start_index, end_index, x, self.relu2, self.bn2, self.conv1)

        self.module_no += 3

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = func_w_index_check(skip_check, self.module_no, start_index, end_index, out, self.conv2)

        self.module_no += 1

        if not self.equalInOut:
            self.module_no += 1
            return torch.add(func_w_index_check(self.skip_check, self.module_no, start_index, end_index, x, self.convShortcut),
                             out), self.module_no
        else:
            return torch.add(x, out), self.module_no


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer_list = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

        self.module_no = 0

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))

        return layers

    def forward(self, x, module_no, start_index,  end_index, skip_check):
        self.module_no = module_no

        a = x
        for i in range(len(self.layer_list)):

            a, self.module_no = self.layer_list[i].forward(a, self.module_no, start_index, end_index, skip_check)
        out = a
        return out, self.module_no


class WideResNet_Sliced(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_Sliced, self).__init__()

        self.module_no = 0
        self.skip_check = [False]  # this is made as a list since list arguments in python fucntions can changed inside

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate).to("cuda")

        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate).to("cuda")
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate).to("cuda")
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
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

        # start index is from which module we should start at

        self.module_no = 0
        out = func_w_index_check(self.skip_check, self.module_no, start_index, end_index, x, self.conv1)
        self.module_no += 1

        out, self.module_no = self.block1(out, self.module_no, start_index, end_index, self.skip_check)
        out, self.module_no = self.block2(out, self.module_no, start_index, end_index, self.skip_check)
        out, self.module_no = self.block3(out, self.module_no, start_index, end_index, self.skip_check)

        out = func_w_index_check(self.skip_check, self.module_no, start_index, end_index, out, self.relu, self.bn1)
        self.module_no += 2

        ##############################################################
        # this we need to figure out of its a function
        if start_index <= self.module_no and end_index > self.module_no:
            out = F.avg_pool2d(out, 8)
            self.module_no += 1
        else:
            pass


        #out = out.view(-1, self.nChannels)
        #return self.fc(out)

        return out


def evaluateModel(m, dataLoader, batchSize):
    correctCount = 0
    N = dataLoader.dataset.data.shape[0]
    for i, (x, y) in enumerate(dataLoader):
        x = x.to("cuda")
        y = y.to("cuda")
        yHat = m(x)
        _, yHat = torch.max(yHat, 1)
        correctCount += (yHat == y).sum().item()
    return correctCount / N


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


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--batchSize", type=int, default=5)
    args.add_argument("--dataset", type=str, default="cifar")
    args.add_argument("--modelFile", type=str, default="../datasets/detection/train/trojan/id-0000/model.pt")
    args = args.parse_args()

    batchSize = args.batchSize
    BATCH_SIZE = batchSize
    DATASET = args.dataset
    MODEL_FILE = args.modelFile

    originalModel = torch.load(MODEL_FILE)
    originalModel.to("cuda")

    if DATASET == "cifar":

        # Do we actually need this? Are means and std dev correct?
        transformToNormalizedTensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transformToNormalizedTensor)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transformToNormalizedTensor)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

        #summary(originalModel, (3, 32, 32))

    elif DATASET == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

        #summary(originalModel, (1, 28, 28))


    else:
        assert False, "ERROR: Wrong dataset"

    print("Evaluating model.....")
    #trainAcc = evaluateModel(originalModel, trainloader, BATCH_SIZE)
    #testAcc = evaluateModel(originalModel, testloader, BATCH_SIZE)
    #print("Train acc=", trainAcc, " Test acc=", testAcc)

    testIter = iter(testloader)

    x, yTrue = testIter.next()
    x = x.to("cuda")
    yTrue = yTrue.to("cuda")
    print("X.size()", x.size())
    print("Y.size()", yTrue.size())
    print("Y", yTrue)

    yPred = originalModel(x)

    print("yPred (one hot)", yPred.size())
    print("yPred", torch.argmax(yPred, dim=1))
    yPred = torch.argmax(yPred, dim=1)
    print("yPred (not one hot)", yPred.size())

    '''
    # Playing with the layers 
    '''

    namedLayers = dict(originalModel.named_modules())

    layerList = flattenTheNN(originalModel)

    breakAndStart = []

    for lIdx in range(len(layerList)):
        if isinstance(layerList[lIdx], nn.ReLU):
            breakAndStart.append([lIdx, lIdx + 1])

    print("DEBUG: 109: breakAndStart: ", breakAndStart)
    modelPairs = []

    NN = WideResNet_Sliced(depth=40, num_classes=10)
    NN.to("cuda")


    state_dict = originalModel.state_dict()
    yPredOriginalModel = originalModel(x)
    x, yTrue = testIter.next()
    x = x.to("cuda")
    yTrue = yTrue.to("cuda")


    for i in range(len(breakAndStart)):

        a = NN.forward(x,0, breakAndStart[i][0])
        a.to("cuda")
        print("=====================================")
        b = NN.forward(a,breakAndStart[i][1], breakAndStart[-1][1])
        print(a.size(), b.size())
        a_new = b.view(-1, NN.nChannels)
        a_n = NN.fc(a_new)
        print(a_n.size())
        if i == 0:
            break

    """
    for i in range(len(breakAndStart)):
        # NN : input ----> breakAndStart[i][0]  =====> breakAndStart[i][1] ------> output
        # NN1: input ----> breakAndStart[i][0]
        # NN2: breakAndStart[i][1] ------> output

        NN1 = nn.Sequential(*layerList[:breakAndStart[i][0] + 1])
        NN2 = nn.Sequential(*layerList[breakAndStart[i][1]:])

        modelPairs.append([NN1, NN2])
    
    print("DEBUG: 119: modelPairs: ")
    for i in range(len(modelPairs)):
        print("--------------------")
        print("modelPairs[", i, "][0]: ", modelPairs[i][0])
        print("modelPairs[", i, "][1]: ", modelPairs[i][1])

    print("DEBUG 132: Checking if NN pairs work")

    yPredOriginalModel = originalModel(x)
    for i in range(len(modelPairs)):
        print("DEBUG 136: ")
        print("NN1  -->", modelPairs[i][0])
        print("NN2  -->", modelPairs[i][1])

        nn1Out = modelPairs[i][0](x)
        print(nn1Out.size())
        nn2Out = modelPairs[i][1](nn1Out)
        print("original out", yPredOriginalModel)
        print("two model out", nn2Out)

    '''
    # Type the individual neuron simulation here
    '''

    print("This accuracy test is not that important at the moment. This is just to keep the code running.")
    accuracy = torch.mean(torch.eq(yTrue, yPred).float())
    print("Accuracy: ", accuracy)

    print("END of Program")
    
"""

    # print(m)