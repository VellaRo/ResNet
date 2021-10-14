from cifarData import CIFAR_dataloaders
from officeData import OFFICE_dataloaders
from imagenetData import IMAGENET_dataloaders

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

    ###IMAGENET
    "IMAGENET_TRAIN":IMAGENET_dataloaders["IMAGENET_TRAIN"],
    "IMAGENET_TEST":IMAGENET_dataloaders["IMAGENET_TEST"],
    "IMAGENET_TRAIN_LVL1":IMAGENET_dataloaders["IMAGENET_TRAIN_LVL1"],
    "IMAGENET_TEST_LVL1":IMAGENET_dataloaders["IMAGENET_TEST_LVL1"],

}

