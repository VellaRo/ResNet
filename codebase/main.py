import torch.nn as nn
from torch.nn.functional import dropout
from torch.optim import Adam
import torchvision.models as models
import argparse


from helpers import get_device
from train import train_model
from cifarData import CIFAR_dataloaders
from officeData import OFFICE_dataloaders 
from eval import eval_model, save_Plot
from losses import edl_digamma_loss , edl_mse_loss , edl_log_loss


def main():
    """
    
    """
    # global
    device = get_device()
    
    
    # TODO:
    # class of dataloder| put all in one big dataloader| add dataloader.num_classes ?
    # describe usage/ methods  
    
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

#########EXPERIMENTS#################
    
    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###

    def defineExperiment(model, criterion_name, optimizer, train_dataloader, num_train_classes, test_dataloader=None, num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, ignoreThreshold = -0.1, uncertainty =False, hierachicalModelPathList=[]):

        ## Set Model directory:
        model_directory = str(model.name) +"/"

        model_directory = model_directory[:-1] + train_dataloader.name[:-5] + "/"
        
        model_directory = model_directory[:-1] + criterion_name+ "/"

        ### CRITERIONS ###
        if criterion_name =="crossEntropy":
            criterion = nn.CrossEntropyLoss()

        #Uncertainty Criterions
        elif criterion_name == "edl_digamma":
            criterion = edl_digamma_loss
            uncertainty =True

        elif criterion_name == "edl_log":
            criterion = edl_log_loss
            uncertainty =True
        
        elif criterion_name == "edl_mse":
            criterion = edl_mse_loss
            uncertainty =True
        else:
            raise Exception("criterion_name not found")

        if pretrained:
            model_directory = model_directory[:-1] + "Pretrained/"

        if train:
            train_acc_hist, train_loss_hist , train_uncertainty_hist = train_model(model, train_dataloader, num_classes =num_train_classes, criterion= criterion, optimizer=optimizer, model_directory= model_directory, device=device , num_epochs=num_epochs, uncertainty=uncertainty)
        
        #Do not use ignoreThreshold
        if ignoreThreshold != -0.1:
            calculate_confusion_Matrix = True
            print("\nignoreThreshold: " + str(ignoreThreshold) + "  eval on: " + test_dataloader.name[:-5] + "\n")
            val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes = num_test_classes, ignoreThreshold=ignoreThreshold, calculate_confusion_Matrix=calculate_confusion_Matrix)
            
        else:
            val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes = num_test_classes)
            


    def runExperiments():
        """
        Runs Experiments specified
        """
        #train coarse Model
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        defineExperiment(model, criterion_name="crossEntropy", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=True, pretrained =True, num_epochs=25, ignoreThreshold = -0.1)
        
        #train fine Model
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        defineExperiment(model, criterion_name="crossEntropy", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TRAIN"], num_train_classes =100 , test_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TEST"], num_test_classes=100 ,train=True, pretrained =True, num_epochs=25, ignoreThreshold = -0.1)

        print("DONE with all expretiments")

    runExperiments()

if __name__ == "__main__":
    main()
    
    
    
