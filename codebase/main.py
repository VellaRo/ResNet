import torch.nn as nn
#from torch.nn.functional import dropout
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import adam

from helpers import get_device
from train import train_model
from dataloadersCollection import dataloaders
 
from eval import eval_model, save_Plot
from losses import edl_digamma_loss , edl_mse_loss , edl_log_loss
from models import resnet18Init

def main():
    """
    
    """
    # global
    device = get_device()
    
    
    # TODO:
    # class of dataloder| put all in one big dataloader


    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###
    # da hier in helper ?
    def defineExperiment(modelList, criterion_name= "crossEntropy", optimizer=None, train_dataloader=dataloaders["CIFAR10_TRAIN"], num_train_classes=0, test_dataloader=dataloaders["CIFAR10_TEST"], num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, uncertaintyThreshold = -0.1, hierarchicalModelPathList = [] ,uncertainty=False, eliminateOpenset= False):
        """
        Buildingblock for Experiments:
            defines evrything that might be needed in the specified experiment when called

        ARGS:   modelList: list of models provided
                cirterion_name: name of defined Loss criterion to use | defined below
                train: if True train the model else only eval
                pretrained: model uses Pretrained weights
                uncertaintyThreshold: between 0,1 , Default: -0,1 marks as not used
                hierarchicalModelPathList = List of path to specified trained Models form modelList 

                optimizer: 
                train_dataloder:
                test_dataloder:
                num_train_classes:
                num_test_classes:
                num_epochs:
        """
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

        #Train
        if train:
            train_acc_hist, train_loss_hist , train_uncertainty_hist = train_model(model, train_dataloader, num_classes =num_train_classes, criterion= criterion, optimizer=optimizer, model_directory= model_directory, device=device , num_epochs=num_epochs, uncertainty=uncertainty)
        
        #Do not use uncertaintyThreshold
        if uncertaintyThreshold != -0.1:

            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, uncertaintyThreshold=uncertaintyThreshold, hierarchicalModelPathList =hierarchicalModelPathList)
        
        #EVAL
        else:
            
            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device, num_classes = num_test_classes, hierarchicalModelPathList =hierarchicalModelPathList , train_dataloader= train_dataloader , test_dataloader =test_dataloader)
            
#########EXPERIMENTS#################
###TRAIN
    def CIFAR100_coarse_AND_fine(train= False, criterion_name = None):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment

            Trains and Evals or only Trains Coarse and fine CIFAR100
        """
    ##train coarse Model
        print("COARSE START\n")
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=train, pretrained =True, num_epochs = 25, uncertaintyThreshold = -0.1)
        print("COARSE END\n")
        
    #train fine Model
        print("FINE START\n")
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["CIFAR100_fine_labels_TRAIN"], num_train_classes =100 , test_dataloader=dataloaders["CIFAR100_fine_labels_TEST"], num_test_classes=100 ,train=train, pretrained =True, num_epochs =25, uncertaintyThreshold = -0.1)
        print("FINE END\n")
    
    def train_ImagenetLVL1(train= False, criterion_name = None):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment

            Trains and Evals or only Trains IMAGENET LVL1
        """
    ##train ImagenetLVL1
        print("IMAGENETLVL1 \n")
        
        model, optimizer = resnet18Init(num_train_classes = 19 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["IMAGENET_LVL1_TRAIN"], num_train_classes =19, test_dataloader=dataloaders["IMAGENET_LVL1_TEST"], num_test_classes=19 ,train=train, pretrained =True, num_epochs = 25, uncertaintyThreshold = -0.1)
        print("IMAGENET \n")
    
    def train_ImagenetNormal(train= False, criterion_name = None):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment

            Trains and Evals or only Trains IMAGENET NORMAL
        """
    ##train Imagenet NORMAL
        print("IMAGENET \n")
        ######################DEBUG##############################
        model, optimizer = resnet18Init(num_train_classes = 1000 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["IMAGENET_TRAIN"], num_train_classes =1000, test_dataloader=dataloaders["IMAGENET_TEST"], num_test_classes=1000 ,train=train, pretrained =True, num_epochs = 25, uncertaintyThreshold = -0.1)
        print("IMAGENET \n")

    def train_ImagenetAnimalsOnly(train= False, criterion_name = None):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment

            Trains and Evals or only Trains IMAGENET ANIMALS ONLY
        """
    ##train Imagenet NORMAL
        print("IMAGENET ANIMALS ONLY\n")
        ######################DEBUG##############################
        model, optimizer = resnet18Init(num_train_classes = 398 , pretrained=True)
        modelList= [model]
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["IMAGENET_ANIMALSONLY_TRAIN"], num_train_classes =398, test_dataloader=dataloaders["IMAGENET_ANIMALSONLY_TEST"], num_test_classes=398 ,train=train, pretrained =True, num_epochs = 25, uncertaintyThreshold = -0.1)
        print("IMAGENET ANIMALS ONLY \n")

###EVAL
    def hierarchicalEval(modelList , optimizer, hierarchicalModelPathList = None,  uncertaintyThresholdRange=[0, 1, 0.05], eliminateOpenset= False):
        """
        ARGS: modelList = List of initialised Models | with for example resnet18Init()
              hierarchicalModelPathList = List of path to specified trained Models form modelList 
              uncertaintyThresholdRange = [start, end, step] | between 0 and 1  
        
        to use: 
                need provide a modelList , and a hierachicalPathList before calling Method
        """
        print("hierarchicalEval START\n")

        start, end, step = uncertaintyThresholdRange
        uncertaintyThresholdList = np.arange(start, end, step).tolist()

        for x in uncertaintyThresholdList:
        
            print("uncertaintyThreshold:\n" + str(x))
                                                                   
            defineExperiment(modelList, criterion_name="crossEntropy", optimizer=optimizer, train_dataloader=dataloaders["CIFAR100_coarse_labels_TRAIN"], num_train_classes =20 , test_dataloader=dataloaders["CIFAR100_coarse_labels_TEST"], num_test_classes=20 ,train=False, pretrained =True, num_epochs =25, uncertaintyThreshold = x,  hierarchicalModelPathList = hierarchicalModelPathList, eliminateOpenset=eliminateOpenset)

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
    #        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader= dataloaders["CIFAR10_TEST"], num_test_classes=10 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
    #        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader= dataloaders["CIFAR90_TEST"], num_test_classes=90 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
    #        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["CIFAR10_TRAIN"], num_train_classes =10 , test_dataloader= dataloaders["CIFAR100_TEST"], num_test_classes=100 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
    
    #####
    def crossDatasetEvaluationOFFICE(train = False, criterion_name = None, uncertaintyThresholdRange = [0, 1, 0.05] ):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment
              uncertaintyThresholdRange = [start, end, step] | between 0 and 1  

        (Trains OFFICE) if train == True
        Does a CrossEvaluation of the OFFICE_DATASETS A,D,W
        """
        
        model_OFFICE, optimizer = resnet18Init(num_train_classes = 31 , pretrained=True)
        modelList_OFFICE= [model_OFFICE]

        start, end, step = uncertaintyThresholdRange
        uncertaintyThresholdList = np.arange(start, end, step).tolist()

        trainingIsDone = False

        #wrong dataset for experiment because share same images over dataset
        for x in uncertaintyThresholdList:
            print("uncertaintyThreshold:" + str(x))

            if train and trainingIsDone == False:
                print("A->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=True, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
                trainingIsDone = True
            else:
                print("A->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->A")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_A_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                
                print("A->D")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->D")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->D")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_D_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    

                print("A->W")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_A_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("D->W")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_D_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)
                print("W->W")
                defineExperiment(modelList_OFFICE, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["OFFICE_W_TRAIN"], num_train_classes =31 , test_dataloader=dataloaders["OFFICE_W_TEST"], num_test_classes=31 ,train=False, pretrained =True, num_epochs = 25, uncertaintyThreshold = x)    
            
            
    
    def runExperiments():
        """
        Runs Experiments specified
        """
        ###### WORKS
        #CIFAR100_coarse_AND_fine(train= True , criterion_name= "crossEntropy")
        ##CIFAR100_coarse_AND_fine(train= False , criterion_name= "edl_log")
#
        ##print("END NORMAL TEST")
        #
        #print("HIERACHIE START")
#        
        train_ImagenetAnimalsOnly(train= True, criterion_name = "crossEntropy")



        modelList =[]
        modelSUPER, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True,
                                        train_dataloader= dataloaders["IMAGENET_LVL1_TRAIN"],
                                         test_dataloader= dataloaders["IMAGENET_LVL1_TEST"])
        
        modelSUB, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True,
                                        train_dataloader= dataloaders["IMAGENET_ANIMALSONLY_TRAIN"],
                                         test_dataloader= dataloaders["IMAGENET_ANIMALSONLY_TRAIN"])
        modelSUB.to(device)
        modelSUPER.to(device)
        # need to append in the same order as hierachical list so in this case [SUPER,SUB]
        modelList.append(modelSUPER) # weniger labels
        modelList.append(modelSUB) #mehr labes
#
        #print("CROSSENTROPY best_model_byUncertainty")
        #hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/best_model_byUncertainty.pth", "./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/best_model_byUncertainty.pth"]
        #hierarchicalEval(modelList=modelList, optimizer =optimizer, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        
        print("CROSSENTROPY best_model_byAcc")
        hierarchicalModelPathList = ["./results/models/ResNet18IMAGENET_LVL1_crossEntropyPretrained/best_model_byAcc.pth", "./results/models/ResNet18IMAGENET_ANIMALSONLY_crossEntropyPretrained/best_model_byAcc.pth"]
        #hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/24.pth", "./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/24.pth"]
        hierarchicalEval(modelList=modelList, optimizer =optimizer, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        ##-------------------#
        ##to low accuracy ~0.15
        #hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_fine_labels_edl_logPretrained/best_model_byUncertainty.pth", "./results/models/ResNet18CIFAR100_coarse_labels_edl_logPretrained/best_model_byUncertainty.pth"]
        ##
        ##hierarchicalEval(modelList=modelList, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        #
        #print("HIERACHIE END")
        ###### STILL TESTING
        ######
     
        #print("OFFICE CROSSDATA")
        #crossDatasetEvaluationOFFICE(train = False, criterion_name = "crossEntropy", uncertaintyThresholdRange = [0.2, 0.9, 0.2] )
        
        #train_ImagenetLVL1(train =True, criterion_name = "crossEntropy")
        #train_ImagenetNormal(train=True, criterion_name= "crossEntropy")
       # hierarchicalEval(modelList=modelList,optimizer=adam ,)

       	print("DONE with all expretiments")

    runExperiments()

if __name__ == "__main__":
    main()
