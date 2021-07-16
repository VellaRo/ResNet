import torch
import torchvision

from torchvision import transforms, datasets
from torch.utils.data import Subset
from random import randrange

transform = transforms.Compose([              
            transforms.Resize(256),                    
            transforms.CenterCrop(224),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
            )])

batch_size = 300
path_A = "./data/Original_images/amazon/images"
path_D = "./data/Original_images/dslr/images"
path_W = "./data/Original_images/webcam/images"
dataset_A = datasets.ImageFolder(path_A, transform = transform)
dataset_D = datasets.ImageFolder(path_D, transform = transform)
dataset_W = datasets.ImageFolder(path_W, transform = transform)


def split_data(dataset):  
    train_idx =[]
    test_idx = []
    for x,data in enumerate(dataset):
        if randrange(4) == 0:
            test_idx.append(x)
        else:
            train_idx.append(x)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    return train_dataset , test_dataset
    
################## A

train_dataset , test_dataset = split_data(dataset_A)

train_dataloader_A = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader_A = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

################## D
train_dataset , test_dataset = split_data(dataset_D)

train_dataloader_D = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader_D = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

################## W
train_dataset , test_dataset = split_data(dataset_W)

train_dataloader_W = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader_W = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

OFFICE_dataloaders = {
    "OFFICE_A_TRAIN": train_dataloader_A,
    "OFFICE_A_TEST": test_dataloader_A,

    "OFFICE_D_TRAIN": train_dataloader_D,
    "OFFICE_D_TEST": test_dataloader_D,
    
    "OFFICE_W_TRAIN": train_dataloader_W,
    "OFFICE_W_TEST": test_dataloader_W,
}
