import torch
import time
import os

from losses import relu_evidence
from helpers import one_hot_embedding , calculate_evidence



def train_model(model, dataloaders, criterion, optimizer, model_directory ,device, num_classes = 10, num_epochs= 1, is_train=True, uncertainty=False):
    print("im using:" + str(device)) # see if using GPU cuda

    since = time.time()
    
    acc_history = []
    loss_history = []
    evidence_history = []

    best_acc = 0.0
    best_evidence = 0.0

    directory = './results/models/' + model_directory
    if not os.path.exists(directory):
        os.makedirs(directory)    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward

            #with uncertainty
            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # save the gradients to _ and predictions in preds
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device)
            
                ############## evidence calculations ##########################
                # U = uncertainty ?
                u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
                
            
            #without uncertainty
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                ############## evidence calculations ##########################
                # U = uncertainty ?
                u, mean_evidence , mean_evidence_succ , mean_evidence_succ = calculate_evidence(preds, labels, outputs, num_classes)
            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        ###me
        epoch_evidence1 =  mean_evidence #total_evidence , ean_evidence_succ ,mean_evidence_fail

        ###me 
        print('Loss: {:.4f} Acc: {:.4f} Uncertainty_mean: {:.4f} Evidence_mean: {:.4f} '.format(epoch_loss, epoch_acc,u.item() ,epoch_evidence1.item()))
        #### herausfinden wie ich uncertainty bekomme und was unterschied zu evidenze ist ???#####
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        if epoch_evidence1 > best_evidence:
            best_evidence = epoch_evidence1
            
        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        evidence_history.append(epoch_evidence1.item())

        # speichert jede Epoche
        torch.save(model.state_dict(), os.path.join(directory, '{0:0=2d}.pth'.format(epoch)))
        print(f"Saved: " + directory + '{0:0=2d}.pth'.format(epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f} Best Evidence: {:4f}'.format(best_acc, best_evidence))
    
    
    return acc_history, loss_history , evidence_history