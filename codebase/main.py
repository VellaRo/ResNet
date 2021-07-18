import torch.nn as nn
#from torch.nn.functional import dropout
import argparse


from helpers import get_device
from train import train_model
from cifarData import CIFAR_dataloaders
from officeData import OFFICE_dataloaders 
from eval import eval_model, save_Plot
from losses import edl_digamma_loss , edl_mse_loss , edl_log_loss
from models import resnet18Init

def main():
    """
    
    """
    # global
    device = get_device()
    
    
    # TODO:
    # class of dataloder| put all in one big dataloader| add dataloader.num_classes ?
    # describe usage/ methods  

#########EXPERIMENTS#################
    
    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###
    # da hier in helper ?
    def defineExperiment(modelList, criterion_name, optimizer, train_dataloader, num_train_classes, test_dataloader=None, num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, ignoreThreshold = -0.1, uncertainty =False, hierachicalModelPathList = []):
        
        model = modelList[0]
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
            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, ignoreThreshold=ignoreThreshold, calculate_confusion_Matrix=calculate_confusion_Matrix,  hierachicalModelPathList =hierachicalModelPathList)
            
        else:
            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, hierachicalModelPathList =hierachicalModelPathList)
            


    def runExperiments():
        """
        Runs Experiments specified
        """
        ##train coarse Model
        
    #    #as experimend method ? | in einer separten klasse ?
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name="crossEntropy", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=False, pretrained =True, num_epochs = 25, ignoreThreshold = -0.1)
       
        #train fine Model
        
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name="crossEntropy", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TRAIN"], num_train_classes =100 , test_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TEST"], num_test_classes=100 ,train=True, pretrained =False, num_epochs =25, ignoreThreshold = -0.1)
    #as experimend method ? | in einer separten klasse ?
    #    # define super model
#try this for coarse because bad eval and high uncertainty even with less classes than fine
    #as experimend method ? | in einer separten klasse ?
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name="edl_log", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=True, pretrained =True, num_epochs = 25, ignoreThreshold = -0.1)
       
        #train fine Model
        
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name="edl_log", optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TRAIN"], num_train_classes =100 , test_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TEST"], num_test_classes=100 ,train=True, pretrained =True, num_epochs =25, ignoreThreshold = -0.1)
    #as experimend method ? | in einer separten klasse ?
        # define super model

        modelList =[]
        modelSUPER, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelSUB, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList.append(modelSUPER)
        modelList.append(modelSUB)
        # mach das automatisch in der ef_experiment methode
        hierachicalModelPathList = ["./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/best_model_byUncertainty.pth", "./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/best_model_byUncertainty.pth"]
        defineExperiment(modelList, criterion_name="crossEntropy", optimizer=optimizer, 
                                train_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 ,
                                test_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,
                                train=False, pretrained =True, num_epochs=25, ignoreThreshold = 0.6,
                                hierachicalModelPathList = hierachicalModelPathList)

        print("DONE with all expretiments")

    runExperiments()

if __name__ == "__main__":
    main()
    
    
    
