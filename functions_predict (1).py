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
import json

# 
def load_model(filepath='checkpoint.pth'):
    '''loads the saved model from the checkpoint,
    returns the saved model'''
    
    checkpoint = torch.load(filepath)
    hidden_layer = checkpoint['hidden_layer']
    structure = checkpoint['model']
    lr = checkpoint['learning_rate']
    model,_,_,_,_ = functions_train.model_setup(structure, hidden_layer, lr)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    
    #using Image to load the picture
    image = Image.open(image)
    
    #resizing, cropping and normalising the loaded picture and transform into a tensor
    img_transformation = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image = img_transformation(image)
    
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model,
    returns top k probalities and classes'''
    
    #preprocessing the image
    image = process_image(image_path).unsqueeze_(0)
   
    #getting the probalities 
    with torch.no_grad():
        
        output = model.forward(image)
        ps = torch.exp(output) 
        top_p, top_class = ps.topk(topk)
        
        #convert top_p to list
        np_top_p = top_p.tolist()[0]
        
        #switch the direction of the key, val to be able to index 
        class_to_idx = {v: k for k, v in model.class_to_idx.items()}
        
        #convert top_class to list
        np_top_class = top_class.tolist()[0]
        
        #pull the top classes we need to index
        top_classes = [class_to_idx[x] for x in np_top_class]
        #(source for indexing and pulling: https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad)
        
    return np_top_p, top_classes

# TODO: Display an image along with the top 5 classes
def sanity_check(image_path, model,categories_json_file, ):
    ''' Returns top k probalities and flower classes for an image '''
   
    with open(categories_json_file, 'r') as json_file:
        cat_to_name = json.load(json_file)
    #Preprocessing and plotting the test picture
    image = process_image(image_path)
    
    #extracting the key of the flower (test image) by splitting the image path
    flower_key = image_path.split('/')[-2]
    #converting the key to the acutal name of the flower from the JSON file
    title = cat_to_name[flower_key]

    #running the prediction/ image classification with our trained model
    probs, top_classes = predict(image_path, model, topk=5)
    # converting the top predicted classes to the acutal name of the flowers from the JSON file
    top_probs_flowers = [cat_to_name[x] for x in top_classes]
    
    return probs, top_probs_flowers 