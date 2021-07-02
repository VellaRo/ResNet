import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def calculate_evidence(preds, labels, outputs, num_classes):
    match = torch.reshape(torch.eq( preds, labels).float(), (-1, 1))
    acc = torch.mean(match)
    evidence = relu_evidence(outputs)
    alpha = evidence + 1
    #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!####
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True) #u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    u = u.mean()
    total_evidence = torch.sum(evidence, 1, keepdim=True)
    mean_evidence = torch.mean(total_evidence)
    mean_evidence_succ = torch.sum(
    torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
    mean_evidence_fail = torch.sum(
    torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

    return u, mean_evidence , mean_evidence_succ , mean_evidence_succ

##### FOR DEBUG 

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
           