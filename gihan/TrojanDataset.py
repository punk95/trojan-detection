import os 
import torch
from torch.utils.data import Dataset
import json

# class GELU(torch.nn.Module):
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return torch.nn.functional.gelu(input)

class TrojanDataset(Dataset):
    

    def __init__(self,path="../datasets/detection/",directory="train",datasetSize = None,train=True, testTrainSplit=0.8):
        self.path = path
        self.directory = directory
        self.datasetSize = datasetSize
        
        
        if self.directory == "train":
            self.path += "train/"
            print("DEBUG: 17: self.path = ",self.path)
            xFileNames = []
            y = []

            
            root, dirs, files = list(os.walk(self.path+"trojan/"))[0]
            for d in dirs:
                if "320" in d: pass
                else:
                    xFileNames.append({"attackSpec":self.path+"trojan/"+d+"/attack_specification.json","info":self.path+"trojan/"+d+"/info.json","model":self.path+"trojan/"+d+"/model.pt"})
                    y.append(True)



            root, dirs, files = list(os.walk(self.path+"clean/"))[0]
            for d in dirs:
                xFileNames.append({"info":self.path+"clean/"+d+"/info.json","model":self.path+"clean/"+d+"/model.pt"})
                y.append(False)


        elif self.directory == "test":
            assert False, "Test set not available"
        elif self.directory == "val":
            assert False, "Not implemented yet"
        else:
            assert False, "Invalid directory"

        self.xFileNames = xFileNames
        self.y = y



        print("Finished creating dataset")


    def __getitem__(self,idx):

        print("opening",self.xFileNames[idx])

        m = torch.load(self.xFileNames[idx]["model"])
        j = json.load(open(self.xFileNames[idx]["info"]))

        print(m)
        print(j)

if __name__=="__main__":
    print("Starting program")
    d = TrojanDataset()

    d.__getitem__(0)
    

