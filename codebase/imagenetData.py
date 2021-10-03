from time import process_time_ns
from numpy.core.fromnumeric import size
import torch
import torchvision
#transform like epretrained model
from torchvision import transforms 
from torch.utils.data import Dataset

#
import glob
from PIL import Image
import os
train_path = './data/imagenet/train'# make a forlder for imagent ./data/imagenet/train 
test_path  ='./data/imagenet/val' #                                  ""
import numpy as np

imageSize = 32
transform = transforms.Compose(
    [transforms.Resize((imageSize,imageSize)),
     #transforms.CenterCrop(224),
     transforms.ToTensor(),
     #transforms.Normalize(                      
     #mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], #[0.485, 0.456, 0.406],                
     #std= [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]#[0.229, 0.224, 0.225]                  
     #)
    ])

batchsize = 64
#lvl 0 is the "base 1000" class 
class ImagenetLvL1(Dataset):
    def __init__(self, transform=None ,train= False):
        self.train= train        
        self.transform = transform

        def makePathlist(startIndex , endIndex):
            classPathsTrain = os.listdir('./data/imagenet/train')
            classPathsVal = os.listdir('./data/imagenet/val')
            print("not sorted")
            pathlistTrain = []
            pathlistVal = []
            for i in range(startIndex, endIndex):
                pathlistTrain.append('./data/imagenet/train/' +str(classPathsTrain[i]))
                pathlistVal.append('./data/imagenet/val/' +str(classPathsVal[i]))
            if train: 
                return pathlistTrain
            else:
                return pathlistVal
        """ 
        def initImageList(pathList= [], label = None):
            print(pathList)
            imageList= np.array([])
            labelList= np.array([])
            
            for i in range(len(pathList)):
                counter =0
                for filename in glob.glob(str(pathList[i]) + '/*.*'): 
                    im=Image.open(filename).convert('RGB')
                    a = transforms.Resize((imageSize, imageSize))
                    im = a(im)
                    im = np.asarray(im)
                    print(np.shape(im))
                    imageList = np.append(imageList, im)
                    labelList = np.append(labelList, label)
        
                    #imageList = np.append(imageList, im)
                    counter = counter +1
                    if counter == 4:
                        break
        """
        def initImageList(pathList= [], labelList = [] ):
            #print(pathList)
            imageList= np.array([])
            #labelList= np.array([])
            
            for i in range(len(pathList)):
                counter =0
                for filename in glob.glob(str(pathList[i]) + '/*.*'): 
                    im=Image.open(filename).convert('RGB')
                    resizeFunc = transforms.Resize((imageSize, imageSize))
                    im = resizeFunc(im)
                    im = np.asarray(im)
                    #print(np.shape(im))
                    imageList = np.append(imageList, im)
                    #labelList = np.append(labelList, labels)
        
                    #imageList = np.append(imageList, im)


            return imageList , np.array(labelList, dtype ="long")

                                                            #400
        imageList, labelList= initImageList(makePathlist(0,399),[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #imageList, labelList = initImageList(makePathlist(4,8),2)
       
        
    
        #imgData = imageList

        #print(imgData.dtype)
        #print("shape image data")
        #print(imgData)
        #print(size(imgData))
        #print(len(imgData))
        #imgData = imageList.reshape(256,3,16,16)
        imgData = imageList
        #(len(imgData)//(3*256*256))
        lenght = len(imgData)
        lenght = lenght //3
        lenght = lenght //imageSize
        lenght = lenght //imageSize
        imgData = imgData.reshape(lenght,3, imageSize ,imageSize) # brauche ich das ?
        #print(np.shape(imgData))
        #print(np.shape(labelList))
        
        #imgData = np.array(imgData)
       # print(np.shape(imgData))
        #self.imgData = imgData.reshape((((len(imgData)//3)//256)),3,16,16)
        print("anderes: ")
        self.imgData = imgData
        #print(self.imgData)
        self.labels = labelList  ## aus einer Datei auslesen
        #np.shape(self.imgData)
        #np.size(self.imgData)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.imgData[index]
        img = np.array( img,dtype='float32')
        class_id = self.labels[index]
        img = torch.tensor(img) 
        class_id = torch.tensor(class_id)
        
        if self.transform != None:
            #img = img.astype(np.uint8)
            #img = Image.fromarray(img, 'RGB')
            #img = Image.fromarray(np.uint8(img), 'RGB')
            ##img = img.numpy()
            img = transforms.ToPILImage()(img)#.to('cpu'))
            img = self.transform(img)
        return img ,class_id

trainImagenetDataLVL1 = ImagenetLvL1(train= True, transform=transform)
testImagenetDataLVL1 = ImagenetLvL1(train= False, transform=transform)
 
train_data_loader_LVL1 = torch.utils.data.DataLoader(
    trainImagenetDataLVL1,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2
)

test_data_loader_LVL1 = torch.utils.data.DataLoader(
    testImagenetDataLVL1,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2
)


imagenet_train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
imagenet_test_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

train_data_loader = torch.utils.data.DataLoader(
    imagenet_train_data,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2
)

test_data_loader = torch.utils.data.DataLoader(
    imagenet_test_data,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2
)

train_data_loader.name = "IMAGENET_TRAIN"
test_data_loader.name = "IMAGENET_TEST"

train_data_loader_LVL1.name = "IMAGENET_TRAIN_LVL1"
test_data_loader_LVL1.name = "IMAGENET_TEST_LVL1"
IMAGENET_dataloaders = {
    "IMAGENET_TRAIN": train_data_loader,
    "IMAGENET_TEST": test_data_loader,
    "IMAGENET_TRAIN_LVL1": train_data_loader_LVL1,
    "IMAGENET_TEST_LVL1": test_data_loader_LVL1,
}