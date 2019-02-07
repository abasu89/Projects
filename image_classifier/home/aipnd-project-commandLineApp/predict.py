import argparse
from time import time, sleep
from os import listdir
import json
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn 
from torch import optim
from torch.autograd import Variable
from PIL import Image

def main():
    in_arg = get_input_args()
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(in_arg.checkpoint)
        
    probs, classes = predict(in_arg.input, model, in_arg.top_k)
    
    flower_names = []
    for c in classes: 
        if c in cat_to_name:
            flower_names.append(cat_to_name[c])
    print ('Flower names: {}'.format(flower_names))
    print ('Respective probabilities: {}'.format(probs))
    return flower_names, probs
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = image.size
    if size[0] > size[1]:
        short, long = size[1], size[0]
    else:
        short, long = size[0], size[1]
    # RESIZE    
    new_short = 256
    new_long = int((256/short)*long)
    img = image.resize((new_short, new_long))
    
    # CENTERCROP
    upper_left_x = (new_long-224)/2
    upper_left_y = (new_short-224)/2
    area = (upper_left_x, upper_left_y, 224+upper_left_x, 224+upper_left_y)
    img = img.crop(area)
    
    # NORMALISE COLOR CHANNELS
    np_image = np.array(img)
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image/255 - means) / stds
    
    # TRANSPOSE IMAGE
    img_tp = np_image.transpose(2, 0, 1)
    
    return img_tp

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    in_arg = get_input_args()
    
    image = Image.open(image_path)
    img = process_image(image)
    imgT = torch.from_numpy(img)
    imgT = imgT.unsqueeze_(0).float()
    
    model.eval()
    
    inputs = Variable(imgT, volatile=True)
    
    if in_arg.gpu:
        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()
            inputs = inputs.cuda()
            print ('executing in CUDA')
        else:
            model.cpu()
            inputs = inputs.cpu()
    else:
        model.cpu()
        inputs = inputs.cpu()
        
    output = model.forward(inputs)
    ps = torch.exp(output)
    
    # topK classes and probabilities 
    probs, indices = torch.topk(ps, topk)
    
    # invert class_to_idx dict
    idx_to_class = {i:c for c,i in model.class_to_idx.items()}
    classes = []
    for i in indices[0]:
        if int(i) in idx_to_class:
            classes.append(idx_to_class[int(i)])
    
    probs = probs.tolist()[0]
    return probs, classes

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    
    return model

def get_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', action='store_true', default=False, help = 'Enter GPU mode')
    parser.add_argument('--category_names', default='./cat_to_name.json', type=str, help = 'Mapping of categories to real names')
    parser.add_argument('--top_k', default=1, type=int, help = 'Top K most likely classes')
    parser.add_argument('--input', default='./flowers/train/101/image_07942.jpg', type=str, help='Path of input image')
    parser.add_argument('--checkpoint', default='./checkpoint.pth', type=str, help='Path of checkpoint to load')

    return parser.parse_args()

if __name__ == "__main__":
    main()