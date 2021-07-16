import glob #https://docs.python.org/3/library/glob.html
import time 
import copy
import torch
from matplotlib import pyplot as plt
import os

from losses import relu_evidence
from helpers import calculate_uncertainty

def eval_model(model, dataloader, model_directory, device, num_classes, ignoreThreshold = -0.1, calculate_confusion_Matrix= False):
    since = time.time()
    
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
    print('saved_model', saved_models)
    
    # goes through all Epochs to find best model after evaluation | best moddel training != best model eval
    for model_path in saved_models:
        print('Loading model', model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        running_corrects = 0
        falsePositiv =0
        truePositiv =0
        flaseNegativ =0
        trueNegativ =0

        classifiedCorrectFN = 0
        classifiedFalseFN   = 0
        
        wasBestModel_byAcc = False
        wasBestModel_byUncertainy = False
        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            u = calculate_uncertainty(preds, labels, outputs, num_classes)
            
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_uncertainty = u.item() 
            
            ##UCERTAINTY IGNORE:::
            
            ignoreThreshold = ignoreThreshold
            #TN: Uncertainty tells us the sample is not a Target and it is Correct
            #FP: Uncertainty tells us the sample is  a Target and it is False
            #FN: Uncertainty tells us the sample is  not a Target and it is Correct
            #TP: Uncertainty tells us the sample is a Target and it is Correct
            
            if calculate_confusion_Matrix:
                for x, label in enumerate(labels):

                    if u.item() >= ignoreThreshold and label.item() <= 9: 
                        flaseNegativ += 1

                        if preds[x] == labels.data[x]:
                            classifiedCorrectFN +=1# but rejected
                        else:
                            classifiedFalseFN +=1 # but rejected

                    if u.item() <ignoreThreshold and label.item()  <= 9: 
                        truePositiv += 1
                    if u.item() >= ignoreThreshold and label.item() > 9:
                        trueNegativ += 1
                    if u.item() <ignoreThreshold and label.item() > 9:
                        falsePositiv += 1
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_byAcc = copy.deepcopy(model.state_dict())
            wasBestModel_byAcc = True

        if epoch_uncertainty < best_uncertainty:
            best_uncertainty = epoch_uncertainty
            best_model_byUncertainty = copy.deepcopy(model.state_dict())
            wasBestModel_byUncertainy = True

        if wasBestModel_byAcc:
                print("\nBestModel_byAcc ..so far RESULTS: \n")
        if wasBestModel_byUncertainy:
                print("\nBestModel_byUncertainty ..so far  RESULTS: \n")
        
        if wasBestModel_byAcc or wasBestModel_byUncertainy:

            print("Results for this epoch: " ) 
            print('Acc: {:.4f}'.format(epoch_acc))
            print('Uncertainty: ' + str(u.item()))
            if calculate_confusion_Matrix:
                print('TP: {:} FP: {:}'.format(truePositiv, falsePositiv))
                print('FN: {:} TN: {:}'.format(flaseNegativ, trueNegativ))
                print('classifiedCorrectFN: {:} \nclassifiedFalseFN: {:}'.format(classifiedCorrectFN, classifiedFalseFN))
       
        acc_history.append(epoch_acc.item())
        uncertainty_history.append(epoch_uncertainty)
        
        print()
    
    torch.save(best_model_byAcc, os.path.join(directory , 'bestmodel_byAcc.pth'))
    print(f"Saved the best model by Accuracy after eval" + directory + 'best_model_byAcc.pth \n')

    torch.save(best_model_byUncertainty, os.path.join(directory , 'best_model_byUncertainty.pth'))
    print(f"Saved the best model by Uncertainty after eval" + directory + 'best_model_byUncertainty.pth \n')
    print('Best Acc: {:4f} Best uncertainty: {:} \n'.format(best_acc , best_uncertainty))
    
    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return acc_history, uncertainty_history



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