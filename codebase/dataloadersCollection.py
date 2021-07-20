from cifarData import CIFAR_dataloaders
from officeData import OFFICE_dataloaders

dataloaders = {
    ######CIFAR
    "CIFAR10_TRAIN": CIFAR_dataloaders["CIFAR10_TRAIN"],
    "CIFAR10_TEST": CIFAR_dataloaders["CIFAR10_TEST"],
    
    "CIFAR100_coarse_labels_TRAIN": CIFAR_dataloaders["CIFAR100_coarse_labels_TRAIN"],
    "CIFAR100_coarse_labels_TEST" : CIFAR_dataloaders["CIFAR100_coarse_labels_TEST"],
    "CIFAR100_fine_labels_TRAIN": CIFAR_dataloaders["CIFAR100_fine_labels_TRAIN"],
    "CIFAR100_fine_labels_TEST" : CIFAR_dataloaders["CIFAR100_fine_labels_TEST"],
    
    "CIFAR90_TEST" : CIFAR_dataloaders["CIFAR90_TEST"],
    
    ###OFFICE
    "OFFICE_A_TRAIN": OFFICE_dataloaders["OFFICE_A_TRAIN"],
    "OFFICE_A_TEST": OFFICE_dataloaders["OFFICE_A_TEST"],

    "OFFICE_D_TRAIN": OFFICE_dataloaders["OFFICE_D_TRAIN"],
    "OFFICE_D_TEST": OFFICE_dataloaders["OFFICE_D_TEST"],
    
    "OFFICE_W_TRAIN": OFFICE_dataloaders["OFFICE_W_TRAIN"],
    "OFFICE_W_TEST": OFFICE_dataloaders["OFFICE_W_TEST"],
}
