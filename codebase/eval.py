import glob #https://docs.python.org/3/library/glob.html
import time 
import copy
import torch
from matplotlib import pyplot as plt
import os

from losses import relu_evidence
from helpers import calculate_evidence

def eval_model(model, dataloaders, model_directory, device, num_classes =10):
    since = time.time()
    
    acc_history = []
    best_acc = 0.0
    best_evidence = 0.0

    best_model = copy.deepcopy(model.state_dict())

    directory = './results/models/' + model_directory
    
    if not os.path.exists(directory):
        os.makedirs(directory)    

    saved_models = glob.glob(directory + '*.pth')
    saved_models.sort()
    print('saved_model', saved_models)

    for model_path in saved_models:
        print('Loading model', model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)

        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            ############## evidence calculations ##########################
            # U = uncertainty ?
            u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        epoch_evidence1 = mean_evidence 
        print('Acc: {:.4f}'.format(epoch_acc))
        print('Evidence: {:.4f}'.format(epoch_evidence1))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

        if epoch_evidence1 > best_evidence:
            best_evidence = epoch_evidence1

        acc_history.append(epoch_acc.item())
        #evidence history ???
        
        print()
    
    torch.save(model.state_dict(), os.path.join(directory , 'bestmodel.pth'))
    print(f"Saved the best model after eval" + directory + 'best_model.pth')

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f} Best Evidenz: {:4f}'.format(best_acc , best_evidence))
    
    return acc_history # evidenz/uncertainty history ?

def save_Plot(train_loss_hist,train_evidence_hist,val_acc_hist,val_acc_hist1 , model_directory):
    
    directory = './results/models/' + model_directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    # save the plots
    plt.figure(0)
    plt.plot(val_acc_hist)
    plt.plot(train_loss_hist)
    plt.savefig(directory + 'trainHistoAccuracyLoss.png')

    plt.figure(1)
    plt.plot(train_evidence_hist)
    plt.savefig(directory + 'trainHistoEvidence.png')
    
    plt.figure(2)
    plt.plot(val_acc_hist1)
    plt.savefig(directory + 'valHistoCIFAR100.png')

    print()
    print("saved TrainHisto" + model_directory)
    print()