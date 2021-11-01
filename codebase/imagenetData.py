from time import process_time_ns
from numpy.core.fromnumeric import size
import torch
from torch.utils.data.dataset import Subset
import torchvision

from torchvision import transforms 
from torch.utils.data import Dataset
from torchvision.datasets import imagenet

#
import glob
from PIL import Image
import os
import shutil                      
import numpy as np

imageSize = 240 # ist das zu groß ?

transform = transforms.Compose(
    [transforms.Resize((imageSize,imageSize)),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(                      
     mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], #[0.485, 0.456, 0.406],                
     std= [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]#[0.229, 0.224, 0.225]                  
     )
    ])

batchsize = 64


def makeLabelFolder(path , labelList, split):
    """
    splits the first 398 ( Animal Classes) into higher Hierachie classes(19)

    """
    folderDirList = os.listdir(path) 
    # need to sort because atfer os.listdir it is not sorted
    folderDirList.sort()
    pathToNewLabels= "./data/imagenet_subSampeled/animalsOnly" + split +"/"

    # make new labels folder ///for imagefolder///
    for i in range(19):
        if os.path.exists(pathToNewLabels + str(i)):
            shutil.rmtree(pathToNewLabels + str(i))
        # os.makedirs(directory) # if exists replace    
        os.makedirs(pathToNewLabels + str(i) , exist_ok=True)
        # die FolderList | hier sind folder  drin

    # The old labels | 1000
    for idxDir ,x in enumerate(folderDirList):
        print(idxDir)
        print(x)
        print("")

        if idxDir >= len(labelList):
            break
        folderPathToImages = path + folderDirList[idxDir] +"/"
        imagesList = os.listdir(folderPathToImages)
        
        #Pictures
        for y in imagesList:
        
            imagesPathList = folderPathToImages + y 
            shutil.copy(imagesPathList, pathToNewLabels + str(labelList[idxDir]))


#makeLabelFolder("./data/imagenet/train/",
#                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                 ,split="train" )

#makeLabelFolder("./data/imagenet/val/",
 #                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #               ,split = "val" )


imagenet_train_dataset = imagenet.ImageFolder(root="./data/imagenet/train/", transform=transform)
imagenet_test_dataset = imagenet.ImageFolder(root="./data/imagenet/val/", transform=transform)
 
imagenet_train_dataloader = torch.utils.data.DataLoader(
    imagenet_train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2
)

imagenet_test_dataloader = torch.utils.data.DataLoader(
    imagenet_test_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2
)
#train labels
targets_train = imagenet_train_dataset.targets
targets_train = list(filter(lambda x: x <398, targets_train))
print( len(targets_train))

#test labels
targets_test = imagenet_test_dataset.targets
targets_test = list(filter(lambda x: x <398, targets_test))
print(len(targets_test))
#dataset animals only

imagenet_animalOnly_dataset_train = Subset( imagenet_train_dataset, targets_train)
print(len(imagenet_animalOnly_dataset_train))
imagenet_animalOnly_dataset_test = Subset( imagenet_test_dataset, targets_test)
print(len(imagenet_animalOnly_dataset_test))

## dataloaders animals only

imagenet_animalsOnly_dataloader_train = torch.utils.data.DataLoader(
    imagenet_animalOnly_dataset_train,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2,
)

#print(len(imagenet_animalsOnly_dataloader_train))
imagenet_animalsOnly_dataloader_test = torch.utils.data.DataLoader(
    imagenet_animalOnly_dataset_test,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2
)
#print(len(imagenet_animalsOnly_dataloader_test))


imagenet_train_datasetLVL1 = imagenet.ImageFolder(root="./data/imagenet_subSampled/animalsOnlytrain", transform= transform)
imagenet_test_datasetLVL1 = imagenet.ImageFolder(root="./data/imagenet_subSampled/animalsOnlyval" , transform= transform)

#print("lvl1:")
#targets = imagenet_train_datasetLVL1.targets
#print(len(list(targets)))

imagenet_train_dataloaderLVL1 = torch.utils.data.DataLoader(
    imagenet_train_datasetLVL1,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2
)


imagenet_test_dataloaderLVL1 = torch.utils.data.DataLoader(
    imagenet_test_datasetLVL1,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2
)


imagenet_train_dataloader.name = "IMAGENET_TRAIN"
imagenet_test_dataloader.name = "IMAGENET_TEST"
imagenet_train_dataloaderLVL1.name = "IMAGENET_LVL1_TRAIN"
imagenet_test_dataloaderLVL1.name = "IMAGENET_LVL1_TEST"
imagenet_animalsOnly_dataloader_train.name = "IMAGENET_ANIMALSONLY_TRAIN"
imagenet_animalsOnly_dataloader_test.name = "IMAGENET_ANIMALSONLY_TEST"

IMAGENET_dataloaders = {
    "IMAGENET_TRAIN": imagenet_train_dataloader,
    "IMAGENET_TEST": imagenet_test_dataloader,
    "IMAGENET_LVL1_TRAIN": imagenet_train_dataloaderLVL1,
    "IMAGENET_LVL1_TEST": imagenet_test_dataloaderLVL1,
    "IMAGENET_ANIMALSONLY_TRAIN" : imagenet_animalsOnly_dataloader_train,
    "IMAGENET_ANIMALSONLY_TEST" : imagenet_animalsOnly_dataloader_test,

}