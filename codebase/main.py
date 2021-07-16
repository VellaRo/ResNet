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
    Example for running : python main.py --crossEntropy
    must have: 
    can have: 
              --pretrained (trained on a model with pretrained weights) # Default pretrained = False
              --uncertrainty (when using an uncertaintyLoss)            # Default uncertainty =False
                    (Example: --mse //NOT INPLEMENTET YET) 
              --epochs (how may epochs for training)                    # Default epochs = 25
              --$LOSS (specifiy Loss Example : --crossEntropy)          # Default --crossEntropy
    """
    #########EXPERIMENTS#################
    
    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###

    # TODO:
    # class of dataloder?? put all in one big dataloader?
    
    # Test if testUncertaintyLoss saves with right directory name
    
    # eval a single Model (Method) | repeat thresholexpreiment with models trained on uncertainty? |
    # add TrainOffice and  evalOffice(experiment)
    # test once more | train and eval ignoreThreshold see if name is right
    
    # make Datasets interchanable in the methods 
    def trainEvalDataset(train =false,train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes = 10, test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes = 10):
        """
        Trains and evals the Dataset
        """
        if train:
            train_acc_hist, train_loss_hist , train_uncertainty_hist = train_model(model, train_dataloader, criterion, optimizer, model_directory, device , num_classes =num_train_classes,  num_epochs=num_epochs ,uncertainty= False)
            print("\ntraining: " str(train_dataloader)+" \n")
        
        print("\nevaluating on: " str(test_dataloader)+" \n")
        val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes=num_test_classes)7

        print("DONE with train Eval Dataset ")

    def testUncertaintyLoss(criterion_name ,uncertainty = True,train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes = 10, test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes = 10):
        """
        Does training and evaluation on a Dataset
        """
        if criterion_name == "edl_digamma":
            model_directory = "Edl_DigammaLoss/"
            criterion = edl_digamma_loss
        elif criterion_name == "edl_log":
            model_directory = "Edl_LogLoss/"
            criterion = edl_log_loss
        elif criterion_name == "edl_mse":
            model_directory = "Edl_Mse_Loss/"
            criterion = edl_mse_loss
        else:
            raise Exception("choose an uncertaintyLoss:  requires \"edl_mse\", \"edl_log\" or \"edl_digamma.\" ")
        
        if args.pretrained:
    
        model_directory = model_directory[:-1] +"Pretrained/"  

        train_acc_hist, train_loss_hist , train_uncertainty_hist = train_model(model, train_dataloader, criterion, optimizer, model_directory, device , num_classes =num_train_classes,  num_epochs=num_epochs ,uncertainty= True)
        val_acc_hist, uncertainty_history = eval_model(model, test_dataloader , model_directory , device, num_classes=num_test_classes)

        print("\nDone with uncertaintyLoss:"+ criterion_name +"\n")


    def testUncertaintyThresholds(ignoreThreshold, train=False, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"], num_train_classes = 10, test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes = 10):
        """
        train= true: Trains on train_dataloader with parameters specified with argument (--pretrained, --$loss ...)
        Evaluates on test_dataloder 
        """
        if train:
            train_acc_hist, train_loss_hist , train_uncertainty_hist = train_model(model, train_dataloader, criterion, optimizer, model_directory, device , num_classes =num_train_classes,  num_epochs=num_epochs ,uncertainty= False, ignoreThreshold = ignoreThreshold)
            print("\ntraining: " str(train_dataloader)+" \n")
        print("\nthreshhold = " +  str(ignoreThreshold) +"evaluation on: "+str(test_dataloader) +"\n")

        val_acc_hist, uncertainty_histry = eval_model(model, test_dataloader, model_directory ,device, num_classes = num_test_classes, ignoreThreshold=ignoreThreshold, calculate_confusion_Matrix=True)
        # saves the histogramms 
        #save_Plot(train_loss_hist,train_uncertainty_hist, val_acc_hist, val_acc_hist1, model_directory)

        print("\n Experint: testUncertaintyThresholds with ingnoreThreshold of:"  + str(ignoreThreshold) + "  DONE \n")

    def runExperiments():
        """
        Runs Experiments specified
        """
        #was okay #testUncertaintyLoss(criterion_name = "edl_log")
        #was okay #testUncertaintyLoss(criterion_name = "edl_mse")
        #was okay #testUncertaintyLoss(criterion_name = "edl_digamma")
        
        #run this again

        testUncertaintyThresholds(ignoreThreshold =0.4 , train = True, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"] , num_train_classes=10,test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes =10)
        testUncertaintyThresholds(ignoreThreshold =0.4 , test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes =10)

        testUncertaintyLoss(criterion_name = "edl_digamma", train = True, train_dataloader=CIFAR_dataloaders["CIFAR10_TRAIN"] , num_train_classes=10,test_dataloader=CIFAR_dataloaders["CIFAR10_TEST"], num_test_classes =10)

        #OFFICE
        #A
        trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_A_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
        #D
        trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_D_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_D_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
        #W
        trainEvalDataset(train =True, train_dataloader=OFFICE_dataloaders["OFFICE_W_TAIN"], num_train_classes =31, test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_A_TEST"], num_test_classes = 31)
        trainEvalDataset(test_dataloader=OFFICE_dataloaders["OFFICE_W_TEST"], num_test_classes = 31)
        
        #testUncertaintyThresholds(ignoreThreshold =0.45, train = False )
        #testUncertaintyThresholds(ignoreThreshold =0.5 , train = False )
        #testUncertaintyThresholds(ignoreThreshold =0.6 , train = False )
        #testUncertaintyThresholds(ignoreThreshold =0.7 , train = False )

        print("DONE with all expretiments")

    #get modelDirectory over parse Arguments and sets Requirements
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epochs", default=25, type=int,
                        help="Desired number of epochs.")
    parser.add_argument("--pretrained", default=False, action="store_true",
                        help="Use a pretrained model.")
    parser.add_argument("--crossEntropy", default=False ,action="store_true",
                        help="Sets loss function to Cross entropy Loss.")                        
    args = parser.parse_args()
    ### Model Parameters
    
    num_epochs = args.epochs
    num_classes = 10 # ouptutclasses of Model to train    
    model = models.resnet18(pretrained=args.pretrained)
    # adapt it to our Data
    model.fc = nn.Linear(512, num_classes)
    device = get_device()
    
   
    
    if args.crossEntropy:

        # Where the model will be saved
        model_directory = "CrossEntropyLoss/"
        criterion = nn.CrossEntropyLoss()

    #elif args.otherCriteron:
        #criterion = otherCriterion()

    else:
            raise Exception("choose an Loss:")

    if args.pretrained:
    
        model_directory = model_directory[:-1] +"Pretrained/"  
        all_parameters = list(model.parameters())
        #we want last layer to have a faster learningrate 
        without_lastlayer =all_parameters[0: len(all_parameters) -2] # -2 weil einmal weiht und einmal Bias vom layer
        #so we extract it
        last_param = model.fc.parameters()

        #passing a nested dict for different learningrate with differen params
        optimizer = Adam([
            {'params': without_lastlayer},
            {'params': last_param, 'lr': 1e-3}
            ], lr=1e-2)
            
        runExperiments()
    # pretrained = False
    else: 
        optimizer = Adam(model.parameters())
        
        runExperiments()


if __name__ == "__main__":
    main()
    
    
    
