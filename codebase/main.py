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
    
    # eval a single Model (Method) | repeat thresholexpreiment with models trained on uncertainty? |
    
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

    def TESTunifiedExperimentMethod(model, criterion_name, optimizer, train_dataloader, num_train_classes, test_dataloader=None, num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, ignoreThreshold = -0.1, uncertainty =False):

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
            val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes = num_test_classes)
        
        else:
            calculate_confusion_Matrix = True
            val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes = num_test_classes, ignoreThreshold=ignoreThreshold, calculate_confusion_Matrix=calculate_confusion_Matrix)

    def runExperiments():
        """
        Runs Experiments specified
        """

       # # will save in wrong directory because "default" name is CIFAR need to correct model directory (Eliminate parse arguments ?| make model direktory dynamicaly in experiments-methods?)
       # ##OFFICE
       # ##A
       # #trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_A_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
       # ##D
       # #trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_D_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
       # ##W
       # #trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_W_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
       # #trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
       # 
       # #testUncertaintyThresholds(ignoreThreshold =0.45, train = False )
       # #testUncertaintyThresholds(ignoreThreshold =0.5 , train = False )
       # #testUncertaintyThresholds(ignoreThreshold =0.6 , train = False )
       # #testUncertaintyThresholds(ignoreThreshold =0.7 , train = False )
#       
        #Test normal Train
        
        #model, optimizer =resnet18Init(pretrained = True, num_train_classes = 10) 
        #TESTunifiedExperimentMethod(model ,train=True ,train_dataloader = CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes = 10, test_dataloader = CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes = 10 , criterion_name="crossEntropy", optimizer =optimizer, pretrained = True, ignoreThreshold = -0.1, uncertainty =False)
        model, optimizer =resnet18Init(pretrained = False, num_train_classes = 31) 
       # TESTunifiedExperimentMethod(model ,train=True ,train_dataloader = OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes = 31, test_dataloader = OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31 , criterion_name="edl_mse", optimizer =optimizer, pretrained = True, ignoreThreshold = 0.8)
       # #still using same modell
        TESTunifiedExperimentMethod(model ,train=True ,train_dataloader = OFFICE_dataloaders["OFFICE_A_TRAIN"], num_train_classes = 31, test_dataloader = OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes = 31 , criterion_name="edl_mse", optimizer =optimizer, pretrained = True, ignoreThreshold = 0.8)

        print("DONE with all expretiments")

    # can remove parser after unified Experiment method| because making problem with model directory
    #get modelDirectory over parse-arguments and sets requirements
   # parser = argparse.ArgumentParser() 
   # parser.add_argument("--epochs", default=25, type=int,
   #                     help="Desired number of epochs.")
   # parser.add_argument("--pretrained", default=False, action="store_true",
   #                     help="Use a pretrained model.")
   # parser.add_argument("--crossEntropy", default=False ,action="store_true",
   #                     help="Sets loss function to Cross entropy Loss.")                        
   # args = parser.parse_args()
    
    #make a setupfunction?? or ut it in unified Experiment method ?
    ### Model Parameters
    
   # num_epochs = args.epochs
   # num_classes = 10 # ouptutclasses of Model to train    
   # model = models.resnet18(pretrained=args.pretrained)
   # model.name = 'ResNet18'
   # # adapt it to our Data
   # model.fc = nn.Linear(512, num_classes)
   # device = get_device()
   # 
   #
   # 
   # if args.crossEntropy:
#
   #     # Where the model will be saved
   #     model_directory = "CrossEntropyLoss/"
   #     criterion = nn.CrossEntropyLoss()
#
   # #elif args.otherCriteron:
   #     #criterion = otherCriterion()
#
   # else:
   #         raise Exception("choose an Loss:")
#
   # if args.pretrained:
   # 
   #     model_directory = model_directory[:-1] +"Pretrained/"  
   #     all_parameters = list(model.parameters())
   #     #we want last layer to have a faster learningrate 
   #     without_lastlayer =all_parameters[0: len(all_parameters) -2] # -2 weil einmal weiht und einmal Bias vom layer
   #     #so we extract it
   #     last_param = model.fc.parameters()
#
   #     #passing a nested dict for different learningrate with differen params
   #     optimizer = Adam([
   #         {'params': without_lastlayer},
   #         {'params': last_param, 'lr': 1e-3}
   #         ], lr=1e-2)
   #         
   #     runExperiments()
   # # pretrained = False
   # else: 
   #     optimizer = Adam(model.parameters())
   #     
   #     runExperiments()
    runExperiments()

if __name__ == "__main__":
    main()
    
    
    
