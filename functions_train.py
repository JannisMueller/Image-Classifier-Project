# import Libaries
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
import dataloader

#defining the inputs for the architectures that are available
structures = {'vgg16':25088,
        'densenet121':1024}

#function for setting up the model and its Hyperparamets
def model_setup(structure, hidden_layer, lr, train_on='gpu'):
    
    '''input: The architecture of the pretrained model, the hyperparamters (hidden_layer, lr) and the 
    Returns: model with the choosen architecure and parameters'''
    if structure == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
    else: print('Application doesnt support this model: Did you mean VGG-16 (vgg16) or Densenet-121 (densenet121)')    
    
    for p in model.parameters():
        p.requires_grad = False
   
        # Replacing the fully connected classifier with a classifier for our images
        classifier = nn.Sequential(nn.Linear(structures[structure], hidden_layer, train_on),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(hidden_layer, 102),
                                   nn.LogSoftmax(dim=1))
    
        model.classifier = classifier
        #defining the Training loss 
        criterion = nn.NLLLoss()
        #definining the optimizer for the parameters of the classifier 
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        device = torch.device("cuda" if torch.cuda.is_available() and train_on == 'gpu' else "cpu")
        model.to(device)

        return model, criterion, optimizer, classifier, lr

# Training and validation loop function
def train_model(model, criterion, optimizer, epochs, path, train_on):
                ''' Trainging the model, returns Training loss, Validation loss and the accuracy of                     model '''
                
                #setting up the training function
                path = path
                Trainloader, Testloader, Validloader, train_data, batch_size =dataloader.dataloader(path)
                steps = 0
                training_loss = 0
                print_every_steps = 40
                device = torch.device("cuda" if torch.cuda.is_available() and train_on == 'gpu' else "cpu")
                
                print('Training and testing of the model with {} Epochs begins'.format(epochs))

                # training loop
                for epoch in range(epochs):
                    for images, labels in Trainloader:
                        steps += 1
                        # Move input and label tensors to the default device
                        images, labels = images.to(device), labels.to(device)
        
                        #setting the gradients to zero
                        optimizer.zero_grad()
        
                        output = model.forward(images)
                        loss_train = criterion(output, labels)
                        loss_train.backward()
                        optimizer.step()

                        training_loss += loss_train.item()

                        # test loop
                        if steps % print_every_steps == 0:
                            validation_loss = 0
                            accuracy_model = 0
                            model.eval()
                            # the gradients should not be updated in the validation loop since we not training
                            with torch.no_grad():
                                for images, labels in Validloader:
                                    # Move input and label tensors to the default device
                                    images, labels = images.to(device), labels.to(device)
                                    output = model.forward(images)
                                    loss = criterion(output, labels)
                                    # summing up total test loss for the model
                                    validation_loss += loss.item()

                                    # Calculating the accuracy of the model
                                    ps = torch.exp(output)
                                    top_p, top_class = ps.topk(1, dim=1)
                                    equals = top_class == labels.view(*top_class.shape)
                                    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                                    # summing up total accuracy for the model
                                    accuracy_model += accuracy

                            print("Epoch: {}/{} | ".format(epoch+1, epochs),
                                  #avg. Training Loss for every print out)
                                  "Training Loss: {:.4f} | ".format(training_loss/print_every_steps), 
                                  #avg. Valid. Loss and Accuracy (summed up loss and accuracy divided by total batches in dataset)
                                  "Validation Loss: {:.4f} | ".format(validation_loss/len(Validloader)), 
                                  "Model Accuracy: {:.4f}".format(accuracy_model/len(Validloader)))
                            #putting the model back in training mode
                            training_loss = 0
                            model.train()
                print('Training and testing of the model is finished')

def test_model(model, train_on):
    '''testing the accuracy of the trained model with the test set, returns accuracy of the model'''
    #setting up the testing function
    total_predictions = 0
    correct_predictions = 0 
    
    path = 'flowers'
    Trainloader, Testloader, Validloader, train_data, batch_size = dataloader.dataloader(path)
    device = torch.device("cuda" if torch.cuda.is_available() and train_on == 'gpu' else "cpu")

    #testing loop
    for images, labels in Testloader:
        with torch.no_grad():
            model.eval()
           
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)

            #calculating the amount of correct predictions
            _, prediction_model = output.max(dim=1)
            correct_predictions += (prediction_model == labels).sum().item()
            total_predictions += labels.size(0)
            
        print('Accuracy on the test set after training: {:.4f}%'.format((correct_predictions/total_predictions)*100))

def save_model(hidden_layer, epochs, structure, classifier, optimizer, model, lr):
    ''' saves the trained model to the checkpoint '''
    path = 'flowers'
    Trainloader, Testloader, Validloader, train_data, batch_size = dataloader.dataloader(path)
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    checkpoint = {'input_size': 25088,
                  'hidden_layer': hidden_layer,
                  'learning_rate': lr,
                  'output_size': 102,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'model': structure,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    print('Model saved to checkpoint')
