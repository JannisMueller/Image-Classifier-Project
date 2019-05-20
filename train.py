'''
Title Image Classifier- Training (Author: Jannis Müller)

Script for training a new neural network on a dataset of images and saving the model to a checkpoint

With the help of command line arguments, the user will be able to define the architecture for the model, the amount of hidden layer ,
the learning rate, the epochs for the training and if the user want to use the GPU for the training and testing.

example command line: python train.py --arch vgg16 --hidden_layer 4096 --learning_rate 0.001 --train_on gpu --epochs 12 

The batch size can be defined in the dataloader function
'''

#import Libaries and functions
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch import optim
import torchvision

from torchvision import datasets, transforms, models
from PIL import Image
import functions_train
import dataloader

#path= 'flowers'

# defining the command line arguments
parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('--arch', action="store", type=str,  help= 'Enter the architecture for the model')
parser.add_argument('--hidden_layer', action="store", type=int, help= 'Enter the amount of hidden layer ')         
parser.add_argument('--learning_rate', action="store", type=float, help= 'Enter the learning rate for the Optimizer')
parser.add_argument('--train_on', action="store", help= 'Enter if the model should run on GPU')
parser.add_argument('--epochs', action="store", type=int, help= 'Enter the amount of epochs for the training loop')
parser.add_argument('--path', action="store", default='flowers',type=str, help= 'Enter the amount of epochs for the training loop')

pa = parser.parse_args()
lr = pa.learning_rate
structure = pa.arch
hidden_layer = pa.hidden_layer
epochs = pa.epochs
train_on = pa.train_on
path = pa.path


#setting up the architecture of model and its Hyperparameters
model, optimizer, criterion, classifier, lr = functions_train.model_setup(structure, hidden_layer, lr, train_on)

#training the model and printing out the Loss and accuracy of the model
functions_train.train_model(model,optimizer, criterion, epochs, path, train_on)

#testing the model
functions_train.test_model(model, train_on)

#save the model to the checkpoint
functions_train.save_model(hidden_layer, epochs, structure, classifier, optimizer, model, lr)
