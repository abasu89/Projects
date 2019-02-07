import argparse
from time import time, sleep
from os import listdir
import preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn 
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict

def main():
    trainloader, testloader, validloader = preprocess.trainloader, preprocess.testloader, preprocess.validloader
    in_arg = get_input_args()
    
    # load pretrained model
    arch = in_arg.arch
    if (arch[0:3] == 'vgg') or (arch[0:8] == 'densenet'):
        net = getattr(models, arch)(pretrained=True)
    else:
        print ('This program only supports VGG and Densenet architectures. The default VGG16 pretrained network will now be loaded.')
        arch = 'vgg16'
        net = getattr(models, arch)(pretrained=True)
    
    # freeze features in pretrained model (no backprop)
    for param in net.parameters():
        param.requires_grad = False
    
    # create new classifier
    if arch[0:3] == 'vgg':
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(net.classifier[0].in_features, in_arg.hidden_units)), 
                                                ('relu1', nn.ReLU(inplace=True)), ('dropout1', nn.Dropout(p=0.5)), 
                                                ('fc2', nn.Linear(in_arg.hidden_units, 102)), ('output', nn.LogSoftmax(dim=1))]))
    elif arch[0:8] == 'densenet':
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(net.classifier.in_features, in_arg.hidden_units)), 
                                                ('relu1', nn.ReLU(inplace=True)), ('dropout1', nn.Dropout(p=0.5)), 
                                                ('fc2', nn.Linear(in_arg.hidden_units, 102)), ('output', nn.LogSoftmax(dim=1))]))

    # assign classifier to pretrained model
    net.classifier = classifier
    print (net.classifier)
    
    # define network parameters
    learnrate = in_arg.learning_rate

    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(net.classifier.parameters(), lr=learnrate) 
    
    # train classifier
    epochs = in_arg.epochs
    print_every = 40
    steps = 0

    for e in range(epochs):
        net.train()
        
        running_loss = 0

        for ii, (images, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            if in_arg.gpu:
                cuda = torch.cuda.is_available()
                if cuda:
                    net = net.cuda()
                    inputs, labels = inputs.cuda(), labels.cuda()
                    print ('EXECUTING IN CUDA')
                else:
                    net = net.cpu()
                    inputs, labels = inputs.cpu(), labels.cpu()
            else:
                net = net.cpu()
                inputs, labels = inputs.cpu(), labels.cpu()
                

            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                net.eval()

                accuracy = 0
                test_loss = 0

                for ii, (images, labels) in enumerate(validloader):

                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)

                    if in_arg.gpu:
                        cuda = torch.cuda.is_available()
                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        else:
                            inputs, labels = inputs.cpu(), labels.cpu()

                    output = net.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                    ps = torch.exp(output).data

                    equality = (labels.data == ps.max(1)[1])

                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                net.train()

    # Save the checkpoint 
    train_datasets = preprocess.train_datasets
    net.class_to_idx = train_datasets.class_to_idx 

    checkpoint = {'features': net.features, 'classifier': net.classifier, 'state_dict': net.state_dict(), 
                  'optimizer': optimizer.state_dict(), 'epochs': epochs, 'class_to_idx': net.class_to_idx,
                  'input_size': net.classifier[0].in_features, 'output_size': 102, 'learning_rate': learnrate, 'arch': in_arg.arch}

    torch.save(checkpoint, in_arg.save_dir)
    

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type = str, default = '/home/workspace/aipnd-project/checkpoint.pth', help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Choose the CNN architecture')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'Set learning rate')
    parser.add_argument('--hidden_units', type=int, default = 512, help = 'Enter number of units in hidden layer')
    parser.add_argument('--epochs', type=int, default = 3, help = 'Enter number of epochs to train over')
    parser.add_argument('--gpu', action='store_true', default=False, help = 'Enter GPU mode')
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
