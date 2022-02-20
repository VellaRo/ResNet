import torch #debug
import torch.nn as nn
#from torch.nn.functional import dropout
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import adam

from helpers import get_device , calculate_uncertainty_all_inputs, append_dropout
from train import train_model
from dataloadersCollection import dataloaders
 
from eval import eval_normal, save_Plot , get_monte_carlo_predictions ,oneImageMC
from losses import edl_digamma_loss , edl_mse_loss , edl_log_loss
from models import resnet18Init

def main():
    print(torch.__version__)

    """

    """
    #DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # global 
    device = get_device() # sollte jetzt gehen , NOtfalls -> torch.device("cuda:0") # 
    
    def make_model_dir(model, train_dataloader, pretrained, num_epochs, criterion_name):

        model_directory = str(model.name) +"/"

        model_directory = model_directory[:-1] + train_dataloader.name[:-5] + "/"
        
        model_directory = model_directory[:-1] + criterion_name+ "/"

        if pretrained == True:
            model_directory = model_directory[:-1] + "pretrained/"
        else:
            model_directory = model_directory[:-1] + "pretrained/"
        
        model_directory = model_directory[:-1] + str(num_epochs) +"Epochs/"
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
        return model_directory

    ### I'will add here future experiments, in the codebase should be everything I used for previous experiments including the Dataloaders ###
    # da hier in helper ?
    def defineExperiment(modelList, criterion_name= "crossEntropy", optimizer=None, train_dataloader=dataloaders["CIFAR10_TRAIN"], num_train_classes=0, test_dataloader=dataloaders["CIFAR10_TEST"], num_test_classes=0 ,  train=False, pretrained =False, num_epochs=25, uncertaintyThreshold = -0.1, hierarchicalModelPathList = [] ,uncertainty=False, eliminateOpenset= False, mc_dropout= False , forward_passes = 3):
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

        if pretrained == True:
            model_directory = model_directory[:-1] + "pretrained/"
        else:
            model_directory = model_directory[:-1] + "pretrained/"
        
        model_directory = model_directory[:-1] + str(num_epochs) +"Epochs/"
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

            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device=device, num_classes = num_test_classes, uncertaintyThreshold=uncertaintyThreshold, hierarchicalModelPathList =hierarchicalModelPathList)
        
        #EVAL
        else:
           
            if mc_dropout:

                mc_dropout_setup(model= model, dropout_rate =0) 

            val_acc_hist, uncertainty_histry = eval_model(modelList, test_dataloader, model_directory ,device=device, num_classes = num_test_classes, hierarchicalModelPathList =hierarchicalModelPathList , train_dataloader= train_dataloader , test_dataloader =test_dataloader)
            
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

    def train_ImagenetAnimalsOnly(train= False, criterion_name = None, mc_dropout = False):
        """
        ARGS: train: if True train the model else only eval
              cirterion_name: name of defined Loss criterion to use | defined in defineExperiment

            Trains and Evals or only Trains IMAGENET ANIMALS ONLY
        """
    ##train Imagenet NORMAL
        print("IMAGENET ANIMALS ONLY\n")
        ######################DEBUG##############################
        model, optimizer = resnet18Init(num_train_classes = 398 , pretrained=True) #DEBUG !!!
        modelList= [model]
        #DEBUG --> under this is pretrained = False
        defineExperiment(modelList, criterion_name=criterion_name, optimizer=optimizer, train_dataloader=dataloaders["IMAGENET_ANIMALSONLY_TRAIN"], num_train_classes =398, test_dataloader=dataloaders["IMAGENET_ANIMALSONLY_TEST"], num_test_classes=398 ,train=train, pretrained = True, num_epochs = 25, uncertaintyThreshold = -0.1, mc_dropout =True)
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

    
            
    def mc_dropout_setup(model, dropout_rate): 
        
            #train_ImagenetAnimalsOnly(train= False, criterion_name = "crossEntropy")
            #"""
            model, optimizer = resnet18Init(num_train_classes = 398 , pretrained=True)
            append_dropout(model, rate= dropout_rate)
    
    def runExperiments():
        # model, optimizer = resnet18Init(num_train_classes = 398 , pretrained=True)
    
        # model_dir = make_model_dir(model, train_dataloader= dataloaders["IMAGENET_ANIMALSONLY_TRAIN"], pretrained=True, num_epochs=25, criterion_name="crossEntropy")

        # print(eval_normal(model= model, dataloader= dataloaders["IMAGENET_ANIMALSONLY_TEST"], model_directory = model_dir))
        ##DEBUG##
        num_train_classes = 10
        model, optimizer = resnet18Init(num_train_classes = num_train_classes , pretrained=True)
        model.load_state_dict(torch.load("./results/models/ResNet18CIFAR10_crossEntropyPretrained/bestmodel_byAcc.pth")) 
        append_dropout(model, rate = 0.3)
        print("dropout appenend")
        #model.load_state_dict(torch.load("./results/models/ResNet18IMAGENET_ANIMALSONLY_crossEntropypretrained75EpochsPretrained/best_model_byAcc.pth")) 
        print("loaded state_dict")
        
       # for i, (image, label) in enumerate(dataloaders["CIFAR10_TEST"]):
       # #for i, (image, label) in enumerate(dataloaders["IMAGENET_ANIMALSONLY_TEST"]):
        
       #     img = image[8].to(get_device())
       #     print("im label"+str(label[8]))
       #     break
        
       # img = img.unsqueeze_(0)
            #input = Variable(image_tensor)
        #img = img.to(device)
        
       # #oneImageMC(50,model,num_train_classes, img)
       # #oneImageMC(20,model,num_train_classes, img)
       # ##oneImageMC(10,model,num_train_classes, img)
       # #oneImageMC(3,model,num_train_classes, img)
        
        #get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],1,model,num_train_classes)
        #get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],2,model,num_train_classes)
        #get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],2,model,num_train_classes)

        #print("3")
        #get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],3,model,num_train_classes)
        #print("4")
        #get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],4,model,num_train_classes)
        # print("3")
        # get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],3,model,num_train_classes)
        # print("5")
        # get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],5,model,num_train_classes)
        # print("20")
        # get_monte_carlo_predictions(dataloaders["IMAGENET_ANIMALSONLY_TEST"],20,model,num_train_classes)

        get_monte_carlo_predictions(dataloaders["CIFAR10_TEST"],50,model,num_train_classes)
        
        #train_ImagenetAnimalsOnly(train = False , criterion_name ="crossEntropy", mc_dropout=True)
        #train_ImagenetAnimalsOnly(train= True, criterion_name = "crossEntropy")

        #model.load_state_dict(torch.load("./results/models/ResNet18IMAGENET_ANIMALSONLY_crossEntropypretrained75EpochsPretrained/best_model_byAcc.pth"))
        #model.eval()
        #model.to(device)
        #dataloader = dataloaders["IMAGENET_ANIMALSONLY_TRAIN"]
        #num_classes = 398
        #calculate_uncertainty_all_inputs(model, dataloader,device, num_classes)
        #"""
        """
        model, optimizer = resnet18Init(num_train_classes = 20 , pretrained=True)
        model.load_state_dict(torch.load("./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/best_model_byAcc.pth"))
        model.eval()
        model.to(device)
        dataloader = dataloaders["CIFAR100_coarse_labels_TEST"]
        num_classes = 20
        calculate_uncertainty_all_inputs(model, dataloader,device, num_classes)
        """
        
        """
        model, optimizer = resnet18Init(num_train_classes = 100 , pretrained=True)
        model.load_state_dict(torch.load("./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/best_model_byAcc.pth"))
        model.eval()
        model.to(device)
        dataloader = dataloaders["CIFAR100_fine_labels_TEST"]
        num_classes = 100
        calculate_uncertainty_all_inputs(model, dataloader,device, num_classes)
        """
        """
        model, optimizer = resnet18Init(num_train_classes = 19 , pretrained=True)
        model.load_state_dict(torch.load("./results/models/ResNet18IMAGENET_LVL1_crossEntropypretrained25EpochsPretrained/best_model_byAcc.pth"))
        model.eval()
        model.to(device)
        dataloader = dataloaders["IMAGENET_LVL1_TRAIN"]
        num_classes = 19
        calculate_uncertainty_all_inputs(model, dataloader,device, num_classes)


        """
        ###Runs Experiments specified
        
        ###### WORKS
        #CIFAR100_coarse_AND_fine(train= True , criterion_name= "crossEntropy")
        ##CIFAR100_coarse_AND_fine(train= False , criterion_name= "edl_log")
#
        ##print("END NORMAL TEST")
        #
        #print("HIERACHIE START")
#        
        #train_ImagenetAnimalsOnly(train= True, criterion_name = "crossEntropy")
        #train_ImagenetLVL1(train= True, criterion_name="crossEntropy")
        """
        #hier 
        modelList =[]
        modelSUPER, optimizer = resnet18Init(num_train_classes = 19 , pretrained=True,
                                        train_dataloader= dataloaders["IMAGENET_LVL1_TRAIN"],
                                         test_dataloader= dataloaders["IMAGENET_LVL1_TEST"])
        
        modelSUB, optimizer = resnet18Init(num_train_classes = 398 , pretrained=True,
                                        train_dataloader= dataloaders["IMAGENET_ANIMALSONLY_TRAIN"],
                                         test_dataloader= dataloaders["IMAGENET_ANIMALSONLY_TEST"])
        modelSUB.to(device)
        modelSUPER.to(device)
        # need to append in the same order as hierachical list so in this case [SUPER,SUB]
        modelList.append(modelSUB) #mehr labes
        modelList.append(modelSUPER) # weniger labels
#
        print("CROSSENTROPY best_model_byA accuracy SUB(398(animals only)) dann super(19(animalsLVL1)) !!!")
        hierarchicalModelPathList = ["./results/models/ResNet18IMAGENET_ANIMALSONLY_crossEntropypretrained75EpochsPretrained/best_model_byAcc.pth", "./results/models/ResNet18IMAGENET_LVL1_crossEntropypretrained25EpochsPretrained/best_model_byAcc.pth"]
        #hierarchicalEval(modelList=modelList, optimizer =optimizer, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0, 1, 0.05])
        
        #print("CROSSENTROPY best_model_byAcc")
       # hierarchicalModelPathList = ["./results/models/ResNet18IMAGENET_LVL1_crossEntropyPretrained/best_model_byAcc.pth", "./results/models/ResNet18IMAGENET_ANIMALSONLY_crossEntropyPretrained/best_model_byAcc.pth"]
        #hierarchicalModelPathList = ["./results/models/ResNet18CIFAR100_fine_labels_crossEntropyPretrained/24.pth", "./results/models/ResNet18CIFAR100_coarse_labels_crossEntropyPretrained/24.pth"]
       
       
       
       #HIER hierarchicalEval(modelList=modelList, optimizer =optimizer, hierarchicalModelPathList = hierarchicalModelPathList , uncertaintyThresholdRange= [0.95, 1, 0.01])


        #hier (end)
        """

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
        #hierarchicalEval(modelList=modelList,optimizer=adam ,)

        print("DONE with all expretiments")

    runExperiments()

if __name__ == "__main__":
    main()
