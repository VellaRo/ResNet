import torch
import time
import os

from losses import relu_evidence
from helpers import one_hot_embedding , calculate_uncertainty


def train_model(model, dataloaders, criterion, optimizer, model_directory ,device, num_classes = 10, num_epochs= 25, uncertainty=False, ignoreThreshold = 0.5):
    print("im using:" + str(device)) # see if using GPU cuda

    since = time.time()
    
    acc_history = []
    loss_history = []
    uncertainty_history = []

    best_acc = -0.1
    best_uncertainty = 10.0

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
                 
                u = calculate_uncertainty(preds, labels, outputs, num_classes)
            
            #without uncertainty_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                u  = calculate_uncertainty(preds, labels, outputs, num_classes)
                
            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        epoch_uncertainty = u.item() 

         
        print('Loss: {:.4f} Acc: {:.4f} Uncertainty_mean: {:.4f} '.format(epoch_loss, epoch_acc, u.item()))
        
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        if epoch_uncertainty < best_uncertainty:
            best_uncertainty = epoch_uncertainty
            
        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        uncertainty_history.append(epoch_uncertainty)

        #saves each Eopch 
        torch.save(model.state_dict(), os.path.join(directory, '{0:0=2d}.pth'.format(epoch)))
        print(f"Saved: " + directory + '{0:0=2d}.pth'.format(epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f} Best Uncertainty: {:4f}'.format(best_acc, best_uncertainty))
    
    
    return acc_history, loss_history , uncertainty_history