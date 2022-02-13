import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def get_device():
    if list(range(torch.cuda.device_count())) !=0 :
        use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def calculate_uncertainty(preds, labels, outputs, num_classes):
    match = torch.reshape(torch.eq( preds, labels).float(), (-1, 1))
    acc = torch.mean(match)
    
    evidence = relu_evidence(outputs)
    #print("outputs")

    #mask = outputs[0][0] >= 0
    #indices = torch.nonzero(mask)
    #print(outputs[indices])
    #print(outputs)

    
    #print("evidence")
    ##print(evidence)
    alpha = evidence + 1
    #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!####
    #squared
    #u = num_classes / torch.sum(alpha * alpha, dim=1, keepdim=True) #u = num_classes / torch.sum(alpha, dim=1, keepdim=True
    # not squared
    u = num_classes / torch.sum(alpha , dim=1, keepdim=True) #u = num_classes / torch.sum(alpha, dim=1, keepdim=True
    
    #DEBUG
    #print("alpha")

    #print(alpha)

    #print("torch.sum(...)")
    #print(torch.sum(alpha, dim=1, keepdim=True))
    ##DEBUG END
   
    u_mean= u.mean()
    #total_evidence = torch.sum(evidence, 1, keepdim=True)
    #mean_evidence = torch.mean(total_evidence)
    #mean_evidence_succ = torch.sum(
    #torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
    #mean_evidence_fail = torch.sum(
    #torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

    return u , u_mean


##### FOR DEBUG 

def calculate_uncertainty_all_inputs(model, dataloader,device, num_classes):#batch size|channels|size(width?)|size(lenght?)
   # inputs_list = torch.empty([256,3,224,224])
   # labels_list = torch.empty([256])
    # DEBUG !!
    import sys

    def predictive_entropy(predictions):
        epsilon = sys.float_info.min
        predictive_entropy = -torch.sum( predictions.mean() * torch.log(predictions.mean() + epsilon,
                axis=-1))

        return predictive_entropy
# DEBUG END
    for (x,(inputs, labels)) in (enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
    #    print(x)
    #    inputs_list = torch.cat([inputs_list, inputs])
    #    labels_list = torch.cat([labels_list, labels])
    
    #print(len(inputs_list))
    #print(labels_list)
        with torch.no_grad():

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        print(preds)
        u, u_mean = calculate_uncertainty(preds,labels, outputs, num_classes)
        predictive_entropy = predictive_entropy(preds)
        #print("u")
        #print(u)
        print("u_mean")
        print(u_mean)
        print("predictive_entropy")
        print(predictive_entropy)
# Print which layer in the model that will compute the gradient

def printActivatedGradients(model):
    print("compute gradients for:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

##### TOOLKIT
def get_paramsListWhereRequires_gradIsTrue(model):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
           