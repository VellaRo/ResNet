import torch
import torch.nn.functional as F
import torch.nn as nn

#DEBUG (For getOneImage())

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
### end

def relu_evidence(y):
    return F.relu(y)

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def get_device():
    if list(range(torch.cuda.device_count())) !=0 :
        use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def calculate_uncertainty(preds, labels, outputs, num_classes):
    match = torch.reshape(torch.eq( preds, labels).float(), (-1, 1))
    acc = torch.mean(match)
    
    evidence = relu_evidence(outputs)
    #print("outputs")

    #mask = outputs[0][0] >= 0
    #indices = torch.nonzero(mask)
    #print(outputs[indices])
    #print(outputs)

    
    #print("evidence")
    ##print(evidence)
    alpha = evidence + 1
    #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!####
    #squared
    #u = num_classes / torch.sum(alpha * alpha, dim=1, keepdim=True) #u = num_classes / torch.sum(alpha, dim=1, keepdim=True
    # not squared
    u = num_classes / torch.sum(alpha , dim=1, keepdim=True) #u = num_classes / torch.sum(alpha, dim=1, keepdim=True
    
    #DEBUG
    #print("alpha")

    #print(alpha)

    #print("torch.sum(...)")
    #print(torch.sum(alpha, dim=1, keepdim=True))
    ##DEBUG END
   
    u_mean= u.mean()
    #total_evidence = torch.sum(evidence, 1, keepdim=True)
    #mean_evidence = torch.mean(total_evidence)
    #mean_evidence_succ = torch.sum(
    #torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
    #mean_evidence_fail = torch.sum(
    #torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

    return u , u_mean


##### FOR DEBUG 

def calculate_uncertainty_all_inputs(model, dataloader,device, num_classes):#batch size|channels|size(width?)|size(lenght?)
   # inputs_list = torch.empty([256,3,224,224])
   # labels_list = torch.empty([256])
    # DEBUG !!
    import sys

    def predictive_entropy(predictions):
        epsilon = sys.float_info.min
        predictive_entropy = -torch.sum( predictions.mean() * torch.log(predictions.mean() + epsilon,
                axis=-1))

        return predictive_entropy
# DEBUG END
    for (x,(inputs, labels)) in (enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
    #    print(x)
    #    inputs_list = torch.cat([inputs_list, inputs])
    #    labels_list = torch.cat([labels_list, labels])
    
    #print(len(inputs_list))
    #print(labels_list)
        with torch.no_grad():

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        print(preds)
        u, u_mean = calculate_uncertainty(preds,labels, outputs, num_classes)
        predictive_entropy = predictive_entropy(preds)
        #print("u")
        #print(u)
        print("u_mean")
        print(u_mean)
        print("predictive_entropy")
        print(predictive_entropy)
# Print which layer in the model that will compute the gradient

def printActivatedGradients(model):
    print("compute gradients for:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

##### TOOLKIT
def get_paramsListWhereRequires_gradIsTrue(model):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

### MC DROPOUT HELPERS

def append_dropout(model, rate=0.2):
            for name, module in model.named_children():
                if len(list(module.children())) > 0:
                    append_dropout(module)
                if isinstance(module, nn.ReLU):
                    new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
                    setattr(model, name, new)

def enable_dropout(model):
                """ Function to enable the dropout layers during test-time """
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()

def getOneImage(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose(
    [transforms.Resize((240,240)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], #[0.485, 0.456, 0.406],                
    std= [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]#[0.229, 0.224, 0.225]                  
    ) 
    ])
    img_normalized =transform(img).float()

    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    img_normalized = img_normalized.to(get_device())
    # print(img_normalized.shape)

    return img_normalized