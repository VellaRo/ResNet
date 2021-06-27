# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#Load data
import torch
import torchvision
import torchvision.transforms as transforms


#transform like epretrained model
from torchvision import transforms

transform = transforms.Compose([            #[1]
transforms.Resize(256),                    #[2]
transforms.CenterCrop(224),                #[3]
transforms.ToTensor(),                     #[4]
transforms.Normalize(                      #[5]
mean=[0.485, 0.456, 0.406],                #[6]
std=[0.229, 0.224, 0.225]                  #[7]
)])

batch_size = 1000

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# CIFAR100

testsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)

testloaderCIFAR100 = torch.utils.data.DataLoader(testsetCIFAR100,            
                            batch_size=batch_size,shuffle=False, num_workers=2)

dataloaders = {
    "train": trainloader,
    "val": testloader,
    "TESTCIFAR100" : testsetCIFAR100,
}


# %%
#Load pretrained Model
import torchvision.models as models

resnet18 = models.resnet18(pretrained= True)

print(resnet18)


# %%
#get a test image

#0 airplane 										
#1 automobile 										
#2 bird 										
#3 cat 										
#4 deer 										
#5 dog 										
#6 frog 										
#7 horse 										
#8 ship 										
#9 truck

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


batch = next(iter(dataloaders['train']))
print(batch[0].shape)
plt.imshow(batch[0][0].permute(1, 2, 0)) # image an der ersten stelle
print(batch[1][0]) # label an der ersten stelle
plt.show()


# %%


# %% [markdown]
# Our pretrained model has 1000 output layers we need to fit them to ur Problem(CIFAR10) so 10 output layers
# 
# 
# We'll change the last linearlayer with model.fc | check the model parameters to get number of inputs
# 
# 
# We also dont want to train the previous layer only the new layer. So we set .requires_grad to False

# %%
def set_parameter_requires_grad(model, feature_extracting = True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

set_parameter_requires_grad(resnet18)


# %%
# inizialise the linear layer
import torch.nn as nn
resnet18.fc = nn.Linear(512, 10)


# %%
# Check which layer in the model that will compute the gradient
for name, param in resnet18.named_parameters():
    if param.requires_grad:
        print(name, param.data)


# %%
#Train the model 

#HELPER####################
def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]
####################################

import time
def train_model(model, dataloaders, criterion, optimizer, device, num_classes = 10, num_epochs=25, is_train=True, uncertainty=False):
    since = time.time()
    
    acc_history = []
    loss_history = []

    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward

            #with uncertainty
            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # save the gradients to _ and predictions in preds
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device)
            #without uncertainty
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        
        torch.save(model.state_dict(), os.path.join('/kaggle/working/', '{0:0=2d}.pth'.format(epoch)))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    
    return acc_history, loss_history


# %%
# Here we only want to update the gradient for the classifier layer that we initialized. 
# settings so only the last layer is trained
from torch.optim import Adam

params_to_update = []
for name,param in resnet18.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
            
optimizer = Adam(params_to_update)


# %%
# Uncertainty LOSSES
import torch.nn.functional as F
###Helper####

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

##############
def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

#The Kullback-Leibler Divergence score, or KL divergence score,
#quantifies how much one probability distribution differs from 
#another probability distribution.
def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    #Sum dirchlet distribution
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    #Sum (number of labers)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # ln (gammafunct) | gammafunc = (n-1)!
    lnB = torch.lgamma(S_alpha) -         torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    # ist das  nicht immer 0? | torch.sum(torch.lgamma(beta)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)
    #ableitung gammafunc(x) / gammafunc(x)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    # sum( dirchlet - 1 *(digamma(Sum_alpha) -(digamma(alpha)) ) 
    # + ???  
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

#???
def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood
#???
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef *         kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

# EQ 4 mit func = digamma
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef *         kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup the loss function
#verschidene kriterien
criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()

#uncertenty auch
#True False

# Train model
train_acc_hist, train_loss_hist = train_model(resnet18, dataloaders["train"], criterion, optimizer, device)

torch.save(state, "./results/ResNet_CrossentropyLoss.pt")
print("Saved: ./results/ResNet_CrossentropyLoss.pt")


# %%
print(len(dataloaders["val"]))


# %%
import glob #https://docs.python.org/3/library/glob.html

def eval_model(model, dataloaders, device):
    since = time.time()
    
    acc_history = []
    best_acc = 0.0

    saved_models = glob.glob('/kaggle/working/' + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)

    for model_path in saved_models:
        print('Loading model', model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('Acc: {:.4f}'.format(epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    
    return acc_history


# %%


val_acc_hist = eval_model(resnet18, test_loader, device)
# val_acc_hist = eval_model(resnet18, test_loaderCIFAR100, device)


# %%
plt.plot(train_acc_hist)
plt.plot(val_acc_hist)
plt.show()


# %%
plt.plot(train_loss_hist)
plt.show()
