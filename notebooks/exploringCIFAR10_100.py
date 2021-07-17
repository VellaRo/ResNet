import torch
from torch.functional import split
from torch.utils.data.dataset import Dataset
import torchvision
#transform like epretrained model
from torchvision import transforms
import numpy as np
import pickle
import os

transform = transforms.Compose([            
            transforms.Resize(256),                    
            transforms.CenterCrop(224),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
            )])

batch_size = 100
#CIFAR10

#0 airplane #1 automobile #2 bird #3 cat #4 deer #5 dog #6 frog #7 horse #8 ship #9 truck

trainsetCIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testsetCIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloaderCIFAR10 = torch.utils.data.DataLoader(trainsetCIFAR10, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloaderCIFAR10 = torch.utils.data.DataLoader(testsetCIFAR10, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# CIFAR100
trainsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
testsetCIFAR100 =  torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)

trainloaderCIFAR100 = torch.utils.data.DataLoader(trainsetCIFAR100,            
                            batch_size=batch_size,shuffle=True, num_workers=2)

testloaderCIFAR100 = torch.utils.data.DataLoader(testsetCIFAR100,            
                            batch_size=batch_size,shuffle=False, num_workers=2)

#trainsetCIFAR100["coarse_labels"]

#for item in trainsetCIFAR100:
#    print(item, type(trainsetCIFAR100[item]))



#def unpickle(file):
#    with open(file, 'rb') as fo:
#        myDict = pickle.load(fo, encoding='latin1')
#    return myDict
#
#trainData = unpickle('./data/cifar-100-python/train')#type of items in each file
#testData = unpickle('./data/cifar-100-python/test')#type of items in each file
#
#
#labelsCoarse_TRAIN = trainData["coarse_labels"]
#labelsCoarse_TEST = testData["coarse_labels"]
#labelsFine_TRAIN = trainData["fine_labels"]
#labelsFine_TEST = testData["fine_labels"]
#
#dataTrain = trainData["data"]
#dataTest = testData["data"]
#
#
#print(type(dataTest)) # numpy array
#print(dataTest.shape)
#print(len(dataTest[0])) # 10000
#
#dataTest = dataTest.reshape(len(dataTest),3,32,32) #transpose ??? changes order
#dataTrain = dataTrain.reshape(len(dataTrain),3,32,32) 
#
#print(dataTest.shape)
#print(type(dataTest)) 
#print(len(dataTest[0])) 
##for x in labelsCoarse_TEST:
##    print(str(labelsCoarse_TEST[x]) +" "+ str(labelsFine_TEST[x])+ "\n")

class DifferentLabelsCIFAR100(Dataset):
    def __init__(self, transform=None ,train= False ,split="coarse_labels"):
                    #root_dir,
        def unpickle(file):
            with open(file, 'rb') as fo:
                myDict = pickle.load(fo, encoding='latin1')
            return myDict

        self.train= train
        try:
            if self.train:
                data = unpickle('./data/cifar-100-python/train')#type of items in each file
            else:
                data = unpickle('./data/cifar-100-python/test')
        except:
            raise Exception("File not fould at path, check if you have downloaded it")
        try:
            #labels
            self.labels = data[split]
        except:
            raise Exception("pls define split (coarse_labels ,fine_labels)")
        imgData = data["data"]
        self.imgData = imgData.reshape(len(imgData),3,32,32) 
    
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.imgData[index]
        class_id = self.labels[index]
        img = torch.tensor(img) 
        class_id = torch.tensor(class_id)
        
        if self.transform != None:
            transToPIL = transforms.ToPILImage()
            img = transToPIL(img)
            img = self.transform(img)
             
        return img ,class_id


        

coarseLabel_Dataset_Train = DifferentLabelsCIFAR100(train= True , transform=transform, split="fine_labels")

trainloaderCIFAR100Coarse = torch.utils.data.DataLoader(coarseLabel_Dataset_Train,            
                            batch_size=batch_size,shuffle=True, num_workers=2)

batch = next(iter(trainloaderCIFAR100Coarse))
for x in range(len(batch[1])):
    print(batch[1][x])
    
trainloaderCIFAR10.name = "CIFAR10_TRAIN"
testloaderCIFAR10.name = "CIFAR10_TEST"
testloaderCIFAR100.name = "CIFAR100_TRAIN"
testloaderCIFAR100.name = "CIFAR100_TEST"
CIFAR_dataloaders = {
    "CIFAR10_TRAIN": trainloaderCIFAR10,
    "CIFAR10_TEST": testloaderCIFAR10,
    "CIFAR100_Train" : trainloaderCIFAR100,
    "CIFAR100_TEST" : testloaderCIFAR100,
#    "CIFAR90_TEST" : testloaderCIFAR90
}


import matplotlib.pyplot as plt

#data= next(iter(CIFAR_dataloaders["CIFAR100_Train"]))
#grid_img = torchvision.utils.make_grid(data[0], nrow=5)
#
#plt.imshow(grid_img.permute(1, 2, 0)  )
#plt.show()