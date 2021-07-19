import torch.nn as nn
#from torch.nn.functional import dropout
import numpy as np

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


    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###
    # da hier in helper ?
    def defineExperiment(modelList, criterion_name, optimizer=None, train_dataloader=None, num_train_classes=0, test_dataloader=None, num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, uncertaintyThreshold = -0.1, uncertainty =False, hierarchicalModelPathList = []):
        
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
        
        #Do not use uncertaintyThreshold
        if uncertaintyThreshold != -0.1:
           # print("\nuncertaintyThreshold: " + str(uncertaintyThreshold) + "  eval on: " + test_dataloader.name[:-5] + "\n")
            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, uncertaintyThreshold=uncertaintyThreshold, hierarchicalModelPathList =hierarchicalModelPathList)
            
        else:
            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, hierarchicalModelPathList =hierarchicalModelPathList)
            
#########EXPERIMENTS#################
###TRAIN
     def CIFAR100_coarse_AND_fine(train= False, criterion_name = None):
    
    ##train coarse Model
        print("COARSE START\n")
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=train, pretrained =True, num_epochs = 25, uncertaintyThreshold = -0.1)
        print("COARSE END\n")
        
    #train fine Model
        print("FINE START\n")
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TRAIN"], num_train_classes =100 , test_dataloader=CIFAR_dataloaders["CIFAR100_fine_labels_TEST"], num_test_classes=100 ,train=train, pretrained =True, num_epochs =25, uncertaintyThreshold = -0.1)
        print("FINE END\n")
    
###EVAL
    def hierarchicalEval(modelList = None, hierarchicalModelPathList = None,  uncertaintyThresholdRange=[0, 1, 0.05] ):
        """
        ARGS: modelList = List of initialised Models | with for example resnet18Init()
              hierarchicalModelPathList = List of path to specified trained Models form modelList 
              uncertaintyThresholdRange = [start, end, step] | between 0 and 1  
        """
        print("hierarchicalEval START\n")
        
        start, end, step = uncertaintyThresholdRange
        uncertaintyThresholdList = np.arange(start, end, step).tolist()

        for x in uncertaintyThresholdList:
        
            print("uncertaintyThreshold:" x)
            defineExperiment(modelList, uncertaintyThreshold = x,
                             hierarchicalModelPathList = hierarchicalModelPathList)
        
        print("hierarchicalEval END\n")
    
    #####
    #wrong dataset for experiment because share same images over dataset
    #def crossDatasetEvaluationCIFAR(train = False, criterion_name = None, uncertaintyThresholdRange = [0, 1, 0.05] ):
    #    model_CIFAR10, optimizer = resnet18Init(num_train_classes = 10 , pretrained=True)
    #    modelList_CIFAR10= [model_CIFAR10]
    #    
    #    model_CIFAR90, optimizer = resnet18Init(num_train_classes = 90 , pretrained=True)
    #    modelList_CIFAR90= [model_CIFAR10]
#
    #    start, end, step = uncertaintyThresholdRange
    #    uncertaintyThresholdList = np.arange(start, end, step).tolist()
#
    #    
    #    for x in uncertaintyThresholdList:
    #        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes=10 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
    #        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader=CIFAR_dataloaders["CIFAR90_TEST"], num_test_classes=90 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
    #        #defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader=CIFAR_dataloaders["CIFAR100_TEST"], num_test_classes=100 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
    
    #####
    def crossDatasetEvaluationOFFICE(train = False, criterion_name = None, uncertaintyThresholdRange = [0, 1, 0.05] ):
        model_OFFICE, optimizer = resnet18Init(num_train_classes = 31 , pretrained=True)
        modelList_OFFICE= [model_OFFICE]

        start, end, step = uncertaintyThresholdRange
        uncertaintyThresholdList = np.arange(start, end, step).tolist()

        trainingIsDone = False

        #wrong dataset for experiment because share same images over dataset
        for x in uncertaintyThresholdList:
            if train and trainingIsDone == False:
                print("A->A")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("A->D")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("A->W")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
                trainingIsDone = True
            else:
                print("A->A")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("A->D")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("A->W")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                
                print("D->A")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->D")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->W")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    

                print("W->A")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->D")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->D")
                defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=OFFICE_dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
            
            
    
    def runExperiments():
        """
        Runs Experiments specified
        """
        #####
        CIFAR100_coarse_AND_fine(train= False , criterion_name= "crossEntropy")
        CIFAR100_coarse_AND_fine(train= False , criterion_name= "edl_log")
        #####
        modelList =[]
        modelSUPER, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelSUB, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList.append(modelSUPER)
        modelList.append(modelSUB)
        
        
        hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/best_model_byUncertainty.pth", "./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/best_model_byUncertainty.pth"]
        
        hierarchicalEval(modelList=modelList, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        #-------------------#
        #hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/best_model_byUncertainty.pth", "./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/best_model_byUncertainty.pth"]
        
        #hierarchicalEval(modelList=modelList, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        ######
        crossDatasetEvaluationOFFICE(train = True, criterion_name = "crossEntropy", uncertaintyThresholdRange = [0, 1, 0.05] )
                               


       	print("DONE with all expretiments")

    runExperiments()

if __name__ == "__main__":
    main()
