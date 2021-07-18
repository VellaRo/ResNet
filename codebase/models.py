import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models

###### INIT MODELS #######

def resnet18Init(num_train_classes, pretrained =False):
    model = models.resnet18(pretrained=pretrained)
    model.name = 'ResNet18'
    # adapt it to our Data
    model.fc = nn.Linear(512, num_train_classes)
    if pretrained:
  
        all_parameters = list(model.parameters())
        #we want last layer to have a faster learningrate 
        without_lastlayer =all_parameters[0: len(all_parameters) -2] # -2 because weight and Bias of the layer
        #so we extract it
        last_param = model.fc.parameters()
        
        #passing a nested dict for different learningrate with differen params
        optimizer = Adam([
            {'params': without_lastlayer},
            {'params': last_param, 'lr': 1e-3}
            ], lr=1e-2)
    else:
        optimizer = Adam(model.parameters())
        
    return model, optimizer