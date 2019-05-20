
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image


path = 'flowers'

def dataloader(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # definfing the transforms for the training set
    data_transforms_train = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomRotation(30),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # definfing the transforms for the validation and testing set
    data_transforms_valid = transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    #Loading the datasets for the training date, the test data and the validation data with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    test_data = datasets.ImageFolder(test_dir,transform=data_transforms_test )
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)

    # Using the three image datasets and the trainforms to define the dataloaders
 

    batch_size = 64

    Trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    Testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    Validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    return Trainloader, Testloader, Validloader, train_data, batch_size


