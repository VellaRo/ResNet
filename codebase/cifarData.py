import torch
import torchvision
#transform like epretrained model
from torchvision import transforms 
from torch.utils.data import Dataset
import pickle
import numpy as np
# transform
transform = transforms.Compose([            
            transforms.Resize(256),                    
            transforms.CenterCrop(224),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
            )])

batch_size = 100

#Custom Datasetclass
    #####DOES NOT PROVIDE A DOWNLOAD##########
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

##KEEP FOR DOWNLOADING DATA
#
#trainsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
#trainloaderCIFAR100 = torch.utils.data.DataLoader(trainsetCIFAR100,            
#                           batch_size=batch_size,shuffle=True, num_workers=2)
#
#testsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)
#testloaderCIFAR100 = torch.utils.data.DataLoader(testsetCIFAR100,            
#                           batch_size=batch_size,shuffle=False, num_workers=2)
#


### Datasets
trainsetCIFAR100_coarse_labels = DifferentLabelsCIFAR100(train= True, transform=transform, split="coarse_labels")

trainsetCIFAR100_fine_labels = DifferentLabelsCIFAR100(train= True, transform=transform, split="fine_labels")

testsetCIFAR100_coarse_labels = DifferentLabelsCIFAR100(train= False, transform=transform, split="coarse_labels")

testsetCIFAR100_fine_labels = DifferentLabelsCIFAR100(train= False, transform=transform, split="fine_labels")

### Loaders
trainloaderCIFAR100_coarse_labels = torch.utils.data.DataLoader(trainsetCIFAR100_coarse_labels,            
                                    batch_size=batch_size,shuffle=True, num_workers=2)

trainloaderCIFAR100_fine_labels = torch.utils.data.DataLoader(trainsetCIFAR100_fine_labels,            
                                    batch_size=batch_size,shuffle=True, num_workers=2)

testloaderCIFAR100_coarse_labels = torch.utils.data.DataLoader(testsetCIFAR100_coarse_labels,            
                                    batch_size=batch_size,shuffle=False, num_workers=2)

testloaderCIFAR100_fine_labels = torch.utils.data.DataLoader(testsetCIFAR100_fine_labels,            
                                    batch_size=batch_size,shuffle=False, num_workers=2)
# CIFAR90
def get_indices(dataset,class_labels_list):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_labels_list:
            pass
        else:
            indices.append(i)
    return indices


testCifar90 = torchvision.datasets.CIFAR100(root='./data', train=False, download =True
                           ,transform=transform)
                           
class_labels_list = [0,1,2,3,4,5,6,7,8,9]
idx = get_indices(testCifar90, class_labels_list)
testloaderCIFAR90 = torch.utils.data.DataLoader(testCifar90,batch_size=batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))


trainloaderCIFAR10.name = "CIFAR10_TRAIN"
testloaderCIFAR10.name = "CIFAR10_TEST"

trainloaderCIFAR100_coarse_labels.name = "CIFAR100_coarse_labels_TRAIN"
testloaderCIFAR100_coarse_labels.name  = "CIFAR100_coarse_labels_TEST"

trainloaderCIFAR100_fine_labels.name = "CIFAR100_fine_labels_TRAIN"
testloaderCIFAR100_fine_labels.name  = "CIFAR100_fine_labels_TEST"

testloaderCIFAR90.name = "CIFAR90_TEST"
CIFAR_dataloaders = {
    "CIFAR10_TRAIN": trainloaderCIFAR10,
    "CIFAR10_TEST": testloaderCIFAR10,
    
    "CIFAR100_coarse_labels_TRAIN":trainloaderCIFAR100_coarse_labels,
    "CIFAR100_coarse_labels_TEST" : testloaderCIFAR100_coarse_labels,
    "CIFAR100_fine_labels_TRAIN": trainloaderCIFAR100_fine_labels,
    "CIFAR100_fine_labels_TEST" : testloaderCIFAR100_fine_labels,
    
    "CIFAR90_TEST" : testloaderCIFAR90,
}

