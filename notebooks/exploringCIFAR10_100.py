import torch
import torchvision
#transform like epretrained model
from torchvision import transforms

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
trainsetCIFAR100 = torchvision.dataset.CIFAR100(root='./data', train=True,download=True, transform=transform)
testsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)

trainloaderCIFAR100 = torch.utils.data.DataLoader(trainsetCIFAR100,            
                            batch_size=batch_size,shuffle=True, num_workers=2)

testloaderCIFAR100 = torch.utils.data.DataLoader(testsetCIFAR100,            
                            batch_size=batch_size,shuffle=False, num_workers=2)


## CIFAR90
#def get_indices(dataset,class_labels_list):
#    indices =  []
#    for i in range(len(dataset.targets)):
#        if dataset.targets[i] in class_labels_list:
#            pass
#        else:
#            indices.append(i)
#    return indices
#
#
#testCifar90 = torchvision.datasets.CIFAR100(root='./data', train=False, download =True
#                           ,transform=transform)
#                           
#class_labels_list = [0,1,2,3,4,5,6,7,8,9]
#idx = get_indices(testCifar90, class_labels_list)
#testloaderCIFAR90 = torch.utils.data.DataLoader(testCifar90,batch_size=batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))

trainloaderCIFAR10.name = "CIFAR10_TRAIN"
testloaderCIFAR10.name = "CIFAR10_TEST"
#testloaderCIFAR90.name = "CIFAR90_TEST"
testloaderCIFAR100.name = "CIFAR100_TRAIN"
testloaderCIFAR100.name = "CIFAR100_TEST"
CIFAR_dataloaders = {
    "CIFAR10_TRAIN": trainloaderCIFAR10,
    "CIFAR10_TEST": testloaderCIFAR10,
    "CIFAR100_Train" : trainloaderCIFAR100,
    "CIFAR100_TEST" : testloaderCIFAR100,
    

#    "CIFAR90_TEST" : testloaderCIFAR90
}

