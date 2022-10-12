from ast import arg
import imp
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn
from argparse import ArgumentParser



def evaluateModel(m,dataLoader,batchSize):
    correctCount = 0
    N = dataLoader.dataset.data.shape[0]
    for i,(x,y) in enumerate(dataLoader):
        x = x.to("cuda")
        y = y.to("cuda")
        yHat = m(x)
        _,yHat = torch.max(yHat,1)
        correctCount += (yHat==y).sum().item()
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

if __name__=="__main__":
    args =  ArgumentParser()
    args.add_argument("--batchSize",type=int,default=5)
    args.add_argument("--dataset",type=str,default="mnist")
    args = args.parse_args()


    batchSize = args.batchSize
    DATASET = args.dataset

    originalModel = torch.load("../datasets/detection/train/trojan/id-0400/model.pt")
    originalModel.to("cuda")
    


    summary(originalModel,(3,32,32))


    if DATASET=="cifar":

        transformToNormalizedTensor = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transformToNormalizedTensor)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transformToNormalizedTensor)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=2)

    elif DATASET=="mnist":
        print("")
        #  

    else:
        assert False, "ERROR: Wrong dataset"


    testIter = iter(testloader)


    x,yTrue = testIter.next()
    x = x.to("cuda")
    yTrue = yTrue.to("cuda")
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
        nn2Out = modelPairs[i][1](nn1Out)
        print("original out",yPredOriginalModel)
        print("two model out",nn2Out)








    '''
    # Type the individual neuron simulation here






    '''









    print("This accuracy test is not that important at the moment. This is just to keep the code running.")
    accuracy = torch.mean(torch.eq(yTrue,yPred).float())
    print("Accuracy: ",accuracy)

    print("END of Program")



    # print(m)
