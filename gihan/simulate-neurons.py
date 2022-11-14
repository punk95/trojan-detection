from ast import arg
from distutils.debug import DEBUG
import imp
from time import time
from unittest import TestLoader
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn
from argparse import ArgumentParser
DEBUG = True
import time

def evaluateModel(m,dataLoader,batchSize,noBatches=None):
    correctCount = 0
    N = dataLoader.dataset.data.shape[0]
    for i,(x,y) in enumerate(dataLoader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        yHat = m(x)
        _,yHat = torch.max(yHat,1)
        correctCount += (yHat==y).sum().item()
        if noBatches!=None:
            if i==noBatches:
                break
            return correctCount/((i+1)*batchSize)

    return correctCount/N








def flattenTheNN(model):
    print("Calling get all children")
    ans = list(model.children())
    print("DEBUG: (13)\t ans: ",ans)
    idx = 0

    while idx<len(ans):
        # print("----->oneIter",idx,ans)
        if len(list(ans[idx].children()))>0:
            ans = ans[:idx]+list(ans[idx].children())+ans[idx+1:]
            # print("----->oneIter(changed)",idx,ans)
        else:
            idx+=1
        
    return ans


def findActivationRanges(model,dataLoader):
    for idx,(miniBatchX,miniBatchY) in enumerate(dataLoader):
        print(" ")

    return None


def getOneDatapointPerLabel(dataLoader):
    return None
    ans = {}
    for i,(x,y) in enumerate(dataLoader):
        # if DEBUG: print("DEBUG: 63: ",x.size(),y.size())
        for idx in range(int(y.size()[0])):
            print("DEBUG: 66: ",i,idx,y[idx],"-->",ans.keys())
            print("DEBUG: 67: ",type(y[0]),type(y[0].detach()),type(y[0].detach().item()))


            if y[idx] not in ans.keys():
                ans[y[idx].item()] = x[idx].item()
            if len(ans.keys())==10:
                break
    return ans


if __name__=="__main__":
    args =  ArgumentParser()
    args.add_argument("--batchSize",type=int,default=5)
    args.add_argument("--dataset",type=str,default="mnist")
    args.add_argument("--modelFile",type=str,default="../datasets/detection/train/trojan/id-0400/model.pt")
    args.add_argument("--device",type=str,default="cuda")
    args = args.parse_args()


    batchSize = args.batchSize
    BATCH_SIZE = batchSize
    DATASET = args.dataset
    MODEL_FILE = args.modelFile
    DEVICE = args.device

    originalModel = torch.load(MODEL_FILE)
    originalModel.to(DEVICE)
    




    if DATASET=="cifar":

        # Do we actually need this? Are means and std dev correct?
        transformToNormalizedTensor = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transformToNormalizedTensor)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transformToNormalizedTensor)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=1)


        summary(originalModel,(3,32,32))

    elif DATASET=="mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=1)
        testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=1)

        summary(originalModel,(1,28,28))


    else:
        assert False, "ERROR: Wrong dataset"


    smallDataset = getOneDatapointPerLabel(testloader)
    if DEBUG: print("DEBUG: 123: important datapoints: ",smallDataset)



    print("Evaluating model.....")
    
    
    trainAcc = evaluateModel(originalModel,trainloader,BATCH_SIZE,noBatches=20)

    t=time.time()
    testAcc = evaluateModel(originalModel,testloader,BATCH_SIZE,noBatches=20)
    print("Time taken (train set): ",time.time()-t)

    t=time.time()
    print("Train acc=",trainAcc," Test acc=",testAcc)
    print("Time taken (test set): ",time.time()-t)

    testIter = iter(testloader)


    totTime=0.0


    while testIter.hasNext():

        x,yTrue = testIter.next()
        x = x.to(DEVICE)
        yTrue = yTrue.to(DEVICE)
        print("X.size()",x.size())
        print("Y.size()",yTrue.size())
        print("Y",yTrue)

        yPred = originalModel(x)



        print("yPred (one hot)",yPred.size())
        print("yPred",torch.argmax(yPred,dim=1))
        yPred = torch.argmax(yPred,dim=1)
        print("yPred (not one hot)",yPred.size())



        '''
        # Playing with the layers 
        '''


        
        namedLayers = dict(originalModel.named_modules())

        layerList = flattenTheNN(originalModel)


        
            


        breakAndStart=[]



        for lIdx in range(len(layerList)):
            if isinstance(layerList[lIdx],nn.ReLU):
                breakAndStart.append([lIdx,lIdx+1])

        print("DEBUG: 109: breakAndStart: ",breakAndStart)
        modelPairs = []

        for i in range(len(breakAndStart)):

            # NN : input ----> breakAndStart[i][0]  =====> breakAndStart[i][1] ------> output
            # NN1: input ----> breakAndStart[i][0]
            # NN2: breakAndStart[i][1] ------> output


            NN1 = nn.Sequential(*layerList[:breakAndStart[i][0]+1])
            NN2 = nn.Sequential(*layerList[breakAndStart[i][1]:])


            modelPairs.append([NN1,NN2])


        print("DEBUG: 119: modelPairs: ")
        for i in range(len(modelPairs)):
            print("--------------------")
            print("modelPairs[",i,"][0]: ",modelPairs[i][0])
            print("modelPairs[",i,"][1]: ",modelPairs[i][1])




        print("DEBUG 132: Checking if NN pairs work")
        
        yPredOriginalModel = originalModel(x)


        for i in range(len(modelPairs)):
            print("DEBUG 136: ")
            print("NN1  -->",modelPairs[i][0])
            print("NN2  -->",modelPairs[i][1])

            nn1Out = modelPairs[i][0](x)
            print(nn1Out.size())


            t=time.time()
            nn2Out = modelPairs[i][1](nn1Out)
            totTime += time.time()-t
            print("totTime=",totTime)
            print("original out",yPredOriginalModel)
            print("two model out",nn2Out)



    print("END:Total time",totTime)




    '''
    # Type the individual neuron simulation here






    '''









    print("This accuracy test is not that important at the moment. This is just to keep the code running.")
    accuracy = torch.mean(torch.eq(yTrue,yPred).float())
    print("Accuracy: ",accuracy)

    print("END of Program")



    # print(m)
