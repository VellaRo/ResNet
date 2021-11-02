import glob #https://docs.python.org/3/library/glob.html
import time 
import copy
import torch
from matplotlib import pyplot as plt
import os
from losses import relu_evidence
from models import resnet18Init
from helpers import calculate_uncertainty

#eliminateOpenset
   

def eval_model(modelList, dataloader, model_directory, device, num_classes, uncertaintyThreshold = -0.1, hierarchicalModelPathList = [], train_dataloader= None , test_dataloader =None, eliminateOpenset=False):
    since = time.time()

    model = modelList[0]
    acc_history = []
    uncertainty_history =[]
    
    best_acc = -0.1
    best_uncertainty = 10.0

    best_model_byAcc = copy.deepcopy(model.state_dict())
    best_model_byUncertainty = copy.deepcopy(model.state_dict())
    directory = './results/models/' + model_directory
    
    if not os.path.exists(directory):
        os.makedirs(directory)    

    saved_models = glob.glob(directory + '*.pth')
    saved_models.sort()

    #def eliminateOpenset():

    def calculate_results():
        ##testin##
        openSetCount = 0
        ##testingEND##
        correctWhileStaySuper  = 0
        correctWhileLeaveSuper = 0
        falseWhileStaySuper    = 0
        falseWhileLeaveSuper   = 0

        running_corrects = 0

        falsePositiv =0
        truePositiv =0
        flaseNegativ =0
        trueNegativ =0

        classifiedCorrectFN = 0
        classifiedFalseFN   = 0
        
        calculate_confusion_Matrix =False

        # hierachicalEval
        if len(hierarchicalModelPathList) >= 2:
            #goes through the hierarche models and saves the labesl, preds, u of evrey Model to a list 
            
            labelsList= [[] for i in range(len(hierarchicalModelPathList))]
            predsList= [[] for i in range(len(hierarchicalModelPathList))]
            uList= [[] for i in range(len(hierarchicalModelPathList))]
            counter =0
            while counter <len(hierarchicalModelPathList):
               

                modelList[counter].load_state_dict(torch.load(hierarchicalModelPathList[counter]))
                modelList[counter].eval()
                modelList[counter].to(device)

                # Iterate over data.
                for inputs,labels in modelList[counter].test_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)  

                    with torch.no_grad():
                    
                        outputs = modelList[counter](inputs)
                        _, preds = torch.max(outputs, 1)
                        running_corrects += torch.sum(preds == labels.data)
                        u , u_mean = calculate_uncertainty(preds, labels, outputs, modelList[counter].num_classes)

                    labelsList[counter].append(labels)
                    predsList[counter].append(preds)   
                    uList[counter].append(u)
                counter +=1   

            # DOES ONLY WORK FOR 2LVL HIERARCHY !!!
            #batches
            for x in range(len(labelsList[0])):
                #attributes in batches
                for y in range(len(labelsList[0][x])):
                    
                    # subModel uncertaintyCheck
                    try:
                        if uList[0][x][y].item() < uncertaintyThreshold:
                            #superModel check preds
                            if predsList[0][x][y] == labelsList[0][x][y].data:
                                correctWhileStaySuper += 1
                            else:
                                falseWhileStaySuper +=1
                        #superModel check preds
                        # einfach einen couter erhöhen für super dann sollte das auch für mehrere funktionieren
                        else:
                            if eliminateOpenset:
                                #wenn wir auf der untersten ebene sind checken wir die threshold wenn abglehn wird sagen 
                                    #wir es gehört nicht zum ziel datenset
                                if uList[1][x][y].item() > uncertaintyThreshold:
                                    openSetCount += 1
                                    ##hier können wir prüfen ob er recht hat in der wir di stelle des bild prüfen
                                        #wir appenden das openset dataset an einem tatsächlichen datenset
                                    ## aber ich glaube das kann auch isoliert voneinander funktioneren,
                                    # da es deterministisch ist und daher nichts an dem ergebniss ändern sollte
                                    # ==> also 1. einmal testen mit closed set sehen wie viele letztedlich rejected werden
                                    #          2. einmal testen mit open set und schauen wie viele er trotzdem annimt
                            if predsList[1][x][y] == labelsList[1][x][y].data:
                                correctWhileLeaveSuper += 1
                            else:
                                falseWhileLeaveSuper += 1
                            
                    except:
                        break
        if eliminateOpenset:
            eliminateOpenset()


        #calculate other results | for "normal" eval
        else:

            # NOTE: NOT TESTED YET !!!!
            # Iterate over data.
            for inputs,labels in dataloader:
                # need iterate throuch labes or input not thorugh dataloers !!!
                inputs = inputs.to(device)
                labels = labels.to(device)  

                with torch.no_grad():

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)

                if uncertaintyThreshold != -0.1:   
                    
                    calculate_confusion_Matrix =True
                    u , u_mean = calculate_uncertainty(preds, labels, outputs, num_classes)
                    #UCERTAINTY IGNORE::                    

                    if calculate_confusion_Matrix:
                        for x in range(len(labels)):
                            #FN: Uncertainty tells us the sample is  not in Target but it is in Target
                            if u[x] >= uncertaintyThreshold and train_dataloader == test_dataloader: 
                                flaseNegativ += 1
                                if preds[x] == labels.data[x]:
                                    classifiedCorrectFN +=1
                                else:
                                    classifiedFalseFN +=1
                            #TP: Uncertainty tells us the sample is in Target and it is in Target
                            if u[x] <uncertaintyThreshold and train_dataloader == test_dataloader : 
                                truePositiv += 1
                           
                            #TN: Uncertainty tells us the sample is not in Target and it is not in Targert
                            if u[x] >= uncertaintyThreshold and  train_dataloader != test_dataloader:
                                trueNegativ += 1
                            #FP: Uncertainty tells us the sample is in Target but it is in not Target
                            if u[x] <uncertaintyThreshold and  train_dataloader != test_dataloader:
                                falsePositiv += 1

                else:
                    u , u_mean = calculate_uncertainty(preds, labels, outputs, num_classes)
            
            if calculate_confusion_Matrix:
                print('TP: {:} FP: {:}'.format(truePositiv, falsePositiv))
                print('FN: {:} TN: {:}'.format(flaseNegativ, trueNegativ))
                print('classifiedCorrectFN: {:} \nclassifiedFalseFN: {:}'.format(classifiedCorrectFN, classifiedFalseFN))
  

        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        epoch_uncertainty = u_mean.item() 
            
            
        return inputs, labels, outputs, preds, running_corrects, u, u_mean, epoch_acc, epoch_uncertainty,  correctWhileStaySuper, correctWhileLeaveSuper, falseWhileStaySuper ,falseWhileLeaveSuper
    
    
    if len(hierarchicalModelPathList) >=2:
        inputs, labels, outputs, preds, running_corrects, u, u_mean, epoch_acc, epoch_uncertainty,  correctWhileStaySuper, correctWhileLeaveSuper, falseWhileStaySuper ,falseWhileLeaveSuper = calculate_results()

        print("correctWhileStaySuper: "+ str(correctWhileStaySuper)+ "  correctWhileLeaveSuper: " + str(correctWhileLeaveSuper)+ "  falseWhileStaySuper: " + str(falseWhileStaySuper) + "  falseWhileLeaveSuper: " +str(falseWhileLeaveSuper) 
            +"\nTOTAL Correct: " +str(correctWhileStaySuper +correctWhileLeaveSuper) +" TOTAL False: " + str(falseWhileLeaveSuper +falseWhileStaySuper)+ " TOTAL: " + str(falseWhileLeaveSuper +falseWhileStaySuper +correctWhileLeaveSuper +correctWhileStaySuper)+"\n" )
     
    else:
        # goes through all Epochs to find best model after evaluation | best model training != best model eval
        for model_path in saved_models:
            print('Loading model', model_path)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)

            inputs, labels, outputs, preds, running_corrects, u, u_mean, epoch_acc, epoch_uncertainty,  correctWhileStaySuper, correctWhileLeaveSuper, falseWhileStaySuper ,falseWhileLeaveSuper = calculate_results()
        
            wasBestModel_byAcc = False
            wasBestModel_byUncertainy = False

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_byAcc = copy.deepcopy(model.state_dict())
                wasBestModel_byAcc = True

            if epoch_uncertainty < best_uncertainty:
                best_uncertainty = epoch_uncertainty
                best_model_byUncertainty = copy.deepcopy(model.state_dict())
                wasBestModel_byUncertainy = True

            if wasBestModel_byAcc: 
                    print("\nBestModel_byAcc ..so far RESULTS:")
            if wasBestModel_byUncertainy:
                    print("BestModel_byUncertainty ..so far  RESULTS: \n")

            if wasBestModel_byAcc or wasBestModel_byUncertainy:
            
                print("Results for this epoch: " ) 
                print('Acc: {:.4f}'.format(epoch_acc))
                print('Uncertainty: ' + str(u_mean.item()))
    
                    
            acc_history.append(epoch_acc.item())
            uncertainty_history.append(epoch_uncertainty)

        torch.save(best_model_byAcc, os.path.join(directory , 'best_model_byAcc.pth'))
        print(f"Saved the best model by Accuracy after eval" + directory + 'best_model_byAcc.pth \n')

        torch.save(best_model_byUncertainty, os.path.join(directory , 'best_model_byUncertainty.pth'))
        print(f"Saved the best model by Uncertainty after eval" + directory + 'best_model_byUncertainty.pth \n')
        print('Best Acc: {:4f} Best uncertainty_mean: {:} \n'.format(best_acc , best_uncertainty))         
                
        print()
    

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return acc_history, uncertainty_history


#Saves history as Plot | not Used/Updated/Tested
def save_Plot(train_loss_hist,train_uncertainty_hist,val_acc_hist,val_acc_hist1 , model_directory):
    
    directory = './results/models/' + model_directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    # save the plots
    plt.figure(0)
    plt.plot(val_acc_hist)
    plt.plot(train_loss_hist)
    plt.savefig(directory + 'trainHistoAccuracyLoss.png')

    plt.figure(1)
    plt.plot(train_uncertainty_hist)
    plt.savefig(directory + 'trainHistoUncertainty.png')
    
    plt.figure(2)
    plt.plot(val_acc_hist1)
    plt.savefig(directory + 'valHistoCIFAR100.png')

    print()
    print("saved TrainHisto" + model_directory)
    print()
