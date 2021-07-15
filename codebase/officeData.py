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
path_A = "./office/data/Original_images/amazon/images"
path_D = "./office/data/Original_images/dslr/images"
path_W = "./office/data/Original_images/webcam/images"
dataset_A = datasets.ImageFolder(path_A, transform = transform)
dataset_D = datasets.ImageFolder(path_D, transform = transform)
dataset_W = datasets.ImageFolder(path_W, transform = transform)


def split_data(dataset):  
    train_idx =[]
    val_idx = []
    for x,data in enumerate(dataset):
        if randrange(4) == 0:
            val_idx.append(x)
        else:
            train_idx.append(x)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset , val_dataset
    
################## A

train_dataset , val_dataset = split_data(dataset_A)

train_dataloader_A = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader_A = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

################## D
train_dataset , val_dataset = split_data(dataset_D)

train_dataloader_D = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader_D = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

################## W
train_dataset , val_dataset = split_data(dataset_W)

train_dataloader_W = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader_W = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

OFFICE_dataloaders = {
    "train_A": train_dataloader_A,
    "val_A": val_dataloader_A,

    "train_D": train_dataloader_D,
    "val_D": val_dataloader_D,
    
    "train_W": train_dataloader_W,
    "val_W": val_dataloader_W,
}
