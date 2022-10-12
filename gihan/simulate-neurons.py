import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn




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








def getAllChildren(model):
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
    batchSize = 1

    m = torch.load("../datasets/detection/train/clean/id-0000/model.pt")
    m.to("cuda")


    summary(m,(3,32,32))


    transformToNormalizedTensor = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transformToNormalizedTensor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transformToNormalizedTensor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=2)

    testIter = iter(testloader)


    x,yTrue = testIter.next()
    x = x.to("cuda")
    yTrue = yTrue.to("cuda")
    print("X.size()",x.size())
    print("Y.size()",yTrue.size())
    print("Y",yTrue)

    yPred = m(x)



    print("yPred (one hot)",yPred.size())
    print("yPred",torch.argmax(yPred,dim=1))
    yPred = torch.argmax(yPred,dim=1)
    print("yPred (not one hot)",yPred.size())



    '''
    # Playing with the layers 
    '''


    
    namedLayers = dict(m.named_modules())
    layerNames = namedLayers.keys()

    layerList = getAllChildren(m)


    
    for l in range(len(layerList)):
        print(l,"-----",layerList[l])
        


    breakAndStart=[]



    for lIdx in range(len(layerNames)):
        if isinstance(namedLayers[layerNames[lIdx]],nn.ReLU):
            breakAndStart.append([lIdx,lIdx+1])


    modelPairs = []

    for i in range(len(breakAndStart)):
        # NN : input ----> breakAndStart[i][0]  =====> breakAndStart[i][1] ------> output
        # NN1: input ----> breakAndStart[i][0]
        # NN2: breakAndStart[i][1] ------> output
        
        # modelPairs.addpend([NN1,NN2])
        print("This is how I make two small models NN1, NN2 by breaking NN ;-)")





    '''
    # Type the individual neuron simulation here






    '''










    accuracy = torch.mean(torch.eq(yTrue,yPred).float())

    print("Accuracy: ",accuracy)

    print("END of Program")



    # print(m)
