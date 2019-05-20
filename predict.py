'''
Title Image Classifier- Prediction (Author: Jannis Müller)

Script for reading in an image and a checkpoint then prints the most likely image class and it's associated probability

With the help of comman line arguments, the user of this script will be able chose the image for the prediction (file path), the top K classes for associated probabilities, the learning rate, the epochs for the training and if the user want to use the GPU for the calculations

example comand line: python predict.py --image_path flowers/test/1/image_06743.jpg --topk 5 --categories_json_file cat_to_name.json
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
import json
from torchvision import datasets, transforms, models
from PIL import Image
import functions_train
import functions_predict
import dataloader

# defining the file path of the images and the checkpoint
path = 'flowers'
filepath = 'checkpoint.pth'

#defining the command line arguments 
parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('--image_path', action="store", default='flowers/test/1/image_06743.jpg', type=str,  help= 'Enter the file path of the image that shoul be predicted')
parser.add_argument('--topk', action="store", type=int, help= 'Enter the top K classes for associated probabilities')         
parser.add_argument('--train_on', action="store", help= 'Enter if the model should run on GPU')
parser.add_argument('--categories_json_file', action="store", default='cat_to_name.json', type=str, help= 'Enter the filepath to the json file with the stored categories')

pa = parser.parse_args()
image_path = pa.image_path
topk = pa.topk
train_on = pa.train_on
categories_json_file = pa.categories_json_file

#open the stored categories from the json file
with open(categories_json_file, 'r') as json_file:
    cat_to_name = json.load(json_file)

#loading the data with the dataloader function
Trainloader, Testloader, Validloader, train_data, batch_size = dataloader.dataloader(path)

#loading the saved model from the checkpoint
model = functions_predict.load_model(filepath)

#preproccessing the image predicting the ouput: Returns top k probalities and flower classes for an image
probs, top_probs_flowers = functions_predict.sanity_check(image_path, model, categories_json_file)

#printing out the top K classes along with associated probabilities
x=0
while x < topk:
    print(" The images shows with a probability of {} a {} ".format(probs[x], top_probs_flowers[x]))
    x += 1
