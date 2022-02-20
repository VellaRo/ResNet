import glob #https://docs.python.org/3/library/glob.html
import time 
import copy
import torch
from matplotlib import pyplot as plt
import os


from losses import relu_evidence
from models import resnet18Init
from helpers import calculate_uncertainty, enable_dropout, get_device , getOneImage

import numpy as np
import torch.nn as nn
import sys
import collections
#eliminateOpenset

def oneImageMC(forward_passes,
                model,
                n_classes, image):
    
    dropout_predictions = np.empty((0,1, n_classes)) #((0,len(dataloader), n_classes)) #10000 images in testdataset 19900
    softmax = nn.Softmax(dim=1)
    device = get_device()
            
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        max_index_output =[]
        model.eval()
        enable_dropout(model)

        model.to(device)

        #image = getOneImage("./data/imagenet_subSampled/animalsOnly398train/1/n01443537_16.JPEG")
        image = image.to(device)
        with torch.no_grad():
            output = model(image)                
            output = softmax(output) # shape (n_samples, n_classes)

            #print("max_value_output")
            #print(max_index_output)
        max_index_output.append(np.argmax(output.cpu().numpy()))
        print("HELOOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(np.shape(max_index_output))
        predictions = np.vstack((predictions, output.cpu().numpy())) # output.cpu().numy()

                #DEBUG
            #_, preds = torch.max(output, 1)
            #label.to(device)
            #preds.to(device)
           # running_corrects += torch.sum(preds == label.data)
            #epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
       # print("epoch acc")
        #print(epoch_acc)
        #DEBUG END
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
    #print(predictions)
    #print(dropout_predictions)
    # Calculating mean across multiple MCD forward passes 
    #sum = np.sum(dropout_predictions, axis=-1)
    #print("sum")
    #print(sum)
    print(max_index_output)
    count_array = []
    for i in range(n_classes):
        count_array.append(max_index_output.count(i))
        print(str(i) + ": " + str(max_index_output.count(i)))
    print("variance conter")
    print(np.var(count_array))
    print("z-score")
    z = (count_array - np.mean(count_array))/ (np.var(count_array) * np.var(count_array ))
    for i in range(len(z)):
        if z[i] < 0:
            z[i] = 0


    print(z)
    print("scaled_z")
    scaled_z = z/np.max(z)
    print(scaled_z)

    max = np.amax(scaled_z)
    max_index = np.argmax(scaled_z)
    
    uncertainty = -1
    for i in range(len(scaled_z)):
        if i != max_index:
            max = max - scaled_z[i]
    uncertainty = max
    print("uncertainty")
    print(uncertainty)
    print("1 - uncertainty")
    print(1 - uncertainty)

    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    #print("meanshape")
    #print(np.shape(mean))
    print("max")
    # max_value = max(mean)
    # max_index = mean.index(max_value)
    max_index = np.argmax(mean)
    max_value = mean[0][max_index]
    print(max_value)
    print(max_index)
    #print(np.shape(dropout_predictions))

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)
    print("var")
    print(np.shape(variance))
    print("var[0][max_index]")
    print(variance[0][max_index])
    print("minvar")
    min_index_var = np.argmin(variance)
    print(min_index_var)
    print(variance[0][min_index_var])

    
    #sorted_dropout_predictions = np.sort(dropout_predictions)
    #top5_sorted_dropout_predictions = sorted_dropout_predictions[:,:,-5]

    #top5_variance = np.var(top5_sorted_dropout_predictions, axis=0)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,) #entropy = -np.sum(mean*np.log(mean), axis=-1)

    # Calculating mutual information across multiple MCD forward passes 
    #mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions),axis=-1), axis=0) # shape (n_samples,)
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),axis=-1), axis=0) # shape (n_samples,)

    std = variance *variance

    var_koifizient = std / mean

    print("var_koifizient")
    var_koifizient_no_NAN = var_koifizient[~np.isnan(var_koifizient)]

    #print(var_koifizient_no_NAN.mean())
    print("mean")
    #print(len(mean))
    #print(mean)
    print(mean.mean())
    print("variance")
    #print(len(variance))
    #print(variance)
    print(variance.mean())
    print()
    print(variance)
    print(np.sort(variance))
    print(np.sum(variance))
    print("scaled(normalize) variance")
    scaled_variance_array = variance/np.max(variance)
    scaled_variance = variance[0][max_index]/np.max(variance)
    print(scaled_variance)
    print("array scaled variance")
    print(scaled_variance_array)
    
    #print("sorted_dropout_predictions")
   # print(sorted_dropout_predictions)
    
    #print("top5")
    #print(top5_variance)
    #print(top5_variance.mean())
    #print("epsilon")
    #print(epsilon)
    print("mutual_info")
    #print(mutual_info)
    print(mutual_info)

    eliminated_Nan_mutual_info = mutual_info[~np.isnan(mutual_info)]
    print(eliminated_Nan_mutual_info)
    print(eliminated_Nan_mutual_info.mean())

def get_monte_carlo_predictions(dataloader,
                                forward_passes,
                                model,
                                n_classes,):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    dataloader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    """

    dropout_predictions = np.empty((0,len(dataloader.dataset), n_classes)) #((0,len(dataloader), n_classes)) #10000 images in testdataset 19900
    softmax = nn.Softmax(dim=1)
    device = get_device()

    # for every model iteration
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        labels = []
        model.eval()
        enable_dropout(model)
        model.to(device)

        # for every batch
        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            for j in label:
                labels.append(label[j])
            with torch.no_grad():
                output = model(image)                
                output = softmax(output) # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy())) # output.cpu().numy()
        
        #stacked prediction of every model iteraation
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
        print("loading...")

    #gets highes prediction from output
    maxPrediction =  np.argmax(dropout_predictions , axis=-1)
    #swap to count (more easily) prediction for each model iteration for the same image 
    maxPrediction=  np.swapaxes(maxPrediction,0,1)

    # get count of each class 
    count_array = np.empty((len(maxPrediction),n_classes))

    for i in range(len(maxPrediction)):
        #print(str(i) +": ")
        #print()
        for j in range(n_classes):
            #print( maxPrediction[i].tolist().count(j))
            count_array[i][j] = maxPrediction[i].tolist().count(j)
            #print(count_array[i][j])
            #print(str(j) + ": " + str(count_array[i][j]))
        #print()

    # calculating z_score
    z_score = np.empty((10000, 10))
    temp_mean = np.mean(count_array, axis=1)
    temp_std = (np.var(count_array, axis=1) * np.var(count_array, axis=1 ))
    for i in range(len(count_array)):
        z_score[i] =  ((count_array[i] - temp_mean[i]) / temp_std[i])
        z_score[i] = z_score[i]/np.max(z_score[i])
    
    # filter out negative scores
    z_score[z_score<0] = 0
    
    # # max and max_index
    max = np.amax(z_score, axis=1)
    max_index = np.argmax(z_score, axis=1)
    
    #calc uncertainty
    certainty = []

    for i in range(len(z_score)):
        temp = max[i]
        for j in range(len(z_score[0])):
            if j != max_index[i] and z_score[i][j] != 0:     
                temp = temp - z_score[i][j]
                #print(temp)
        certainty.append(temp)
    certainty = np.array(certainty)
    certainty[certainty<0] = 0
    uncertainty = 1 - certainty
    print("uncertainty")
    print(uncertainty)
    
    print(np.shape(uncertainty))
    print("mean")
    print(np.mean(uncertainty))
    for i in range(20,40):
        print("uncertainty: " +str(uncertainty[i]))
        print(z_score[i])
        print(count_array[i])
        print("predicted label" + str(max_index[i]))
        print("acutal label" +str(labels[i]))
    return uncertainty , labels , max_index, max

def eval_ALL():
    #TODO:
        #eval_normal
        # --> get accuracy 
        #get_monte_carlo_predictions to uncertainty
        #eval hierachie

    pass
    return

def hierachicalEval(uncertaintyThreshold, accuracyThreshold ): #things from montecarlo
    
    uncertaintyList0, labels0 ,preds0, predsAccuracyLists0 = get_monte_carlo_predictions() #0 is the model with more target classes
    uncertaintyList1, labels1 ,preds1, predsAccuracyLists1 = get_monte_carlo_predictions()
    
    uncertaintyLists = [uncertaintyList0, uncertaintyList1]
    labelsList= [labels0, labels1]
    predsList= [preds0, preds1]
    
    truePositiveAcc  = 0
    falsePositiveAcc = 0
    trueNegativeAcc = 0
    falseNegativeAcc = 0
    wrongClassified_MoreClassesAcc = 0
    wrongClassified_LessClassesAcc = 0

    truePositiveUnc  = 0
    falsePositiveUnc = 0
    trueNegativeUnc = 0
    falseNegativeUnc = 0
    wrongClassified_MoreClassesUnc = 0
    wrongClassified_LessClassesUnc = 0

# DOES ONLY WORK FOR 2LVL HIERARCHY !!!

    ## Hierachical eval with uncertainty as measure

    for i in range(len(uncertaintyLists[0])):
        if uncertaintyList[0][i] < uncertaintyThreshold:
            #we are staying in the model with more target classes
            if predsList[0][i] == labelsList[0][i]:
                truePositiveUnc = +=1 #was predicted to be in Dataset and IS in Dataset
                correctClassified_MoreClassesUnc = +=1
            else:
                falsePositiveUnc = +=1 #was predicted to be in Dataset but IS NOT in Dataset
                wrongClassified_MoreClassesUnc = +=1
        elif uncertaintyList[1][i] < uncertaintyThreshold:    
                if predsList[1][i] == labelsList[1][i]:
                    truePositiveUnc += 1 #was predicted to be in Dataset and IS in Dataset
                    correctClassified_LessClassesUnc = +=1
                else:
                    falsePositiveUnc += 1 #was predicted to be in Dataset but IS NOT in Dataset
                    wrongClassified_LessClassesUnc = +=1
        else: # uncertainty to high in model(with less classes) --> All should NOT be in Dataset
            if predsList[1][i] == labelsList[1][i]:
                falseNegativeUnc +=1 #was predicted NOT to be in Dataset but IS in Dataset
                wrongClassified_LessClassesUnc = +=1
            else:
                trueNegativeUnc +=1 #was predicted NOT to be in Dataset and IS NOT in Dataset
                correctClassified_LessClassesUnc = +=1
    
    ## Hierachical eval with accuracy as measure
    
    for i in range(len(accuracyLists[0])):
        if predsAccuracyLists[0][i] < accuracyThreshold:
            #we are staying in the model with more target classes
            if predsList[0][i] == labelsList[0][i]:
                truePositiveAcc = +=1 #was predicted to be in Dataset and IS in Dataset
                correctClassified_MoreClassesAcc = +=1
            else:
                falsePositiveAcc = +=1 #was predicted to be in Dataset but IS NOT in Dataset
                wrongClassified_MoreClassesAcc = +=1
        elif predsAccuracyLists[1][i] < accuracyThreshold:    
                if predsList[1][i] == labelsList[1][i]:
                    truePositiveAcc += 1 #was predicted to be in Dataset and IS in Dataset
                    correctClassified_LessClassesAcc = +=1
                else:
                    falsePositiveAcc += 1 #was predicted to be in Dataset but IS NOT in Dataset
                    wrongClassified_LessClassesAcc = +=1
        else: # predsAccuracy to high in model(with less classes) --> All should NOT be in Dataset
            if predsList[1][i] == labelsList[1][i]:
                falseNegativeAcc +=1 #was predicted NOT to be in Dataset but IS in Dataset
                wrongClassified_LessClassesAcc = +=1
            else:
                trueNegativeAcc +=1 #was predicted NOT to be in Dataset and IS NOT in Dataset
                correctClassified_LessClassesAcc = +=1

    return truePositiveAcc, falsePositiveAcc, trueNegativeAcc, falseNegativeAcc, wrongClassified_MoreClassesAcc, wrongClassified_LessClassesAcc,       truePositiveUnc, falsePositiveUnc, trueNegativeUnc, falseNegativeUnc, wrongClassified_MoreClassesUnc, wrongClassified_LessClassesUnc     

def eliminateOpenset():
    pass

def eval_normal(model, dataloader, model_directory):
    device = get_device()
    since = time.time()

    directory = './results/models/' + model_directory
        
    if not os.path.exists(directory):
        os.makedirs(directory)   
    
    saved_models = glob.glob(directory + '*.pth')
    saved_models.sort()
    
    model.eval()
    model.to(device)
    
    best_acc = 0.0
    for model_path in saved_models:
        print('Loading model', model_path)
        model.load_state_dict(torch.load(model_path))    

        running_corrects = 0
        epoch_acc = 0.0

        # Iterate over data.
        for inputs,labels in dataloader:
            # need iterate throuch labes or input not thorugh dataloers !!!
            inputs = inputs.to(device)
            labels = labels.to(device)  

            with torch.no_grad():

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
        # save Best Model by accuracy 
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_byAcc = copy.deepcopy(model.state_dict())
            wasBestModel_byAcc = True

        torch.save(best_model_byAcc, os.path.join(directory , 'best_model_byAcc.pth'))
        print(f"Saved the best model by Accuracy after eval so far" + directory + 'best_model_byAcc.pth \n')

        # printing Results
        print("Results for this epoch: " ) 
        print('Acc: {:.4f}'.format(epoch_acc))

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return epoch_acc

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
