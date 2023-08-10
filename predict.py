import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from train import build_model 

def get_args():
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--input_image", type=str, help="path to the input image",default='/content/drive/MyDrive/flowers/test/1/image_06743.jpg')
    parser.add_argument("--checkpoint_vgg", type=str, help="path to the checkpoint file",
                        default='/content/drive/MyDrive/checkpoint.pth')
    parser.add_argument("--top_k", type=int, default=5, help="number of top predictions to return")
    parser.add_argument("--category_names", type=str, default='/content/cat_to_name.json',
                        help="path to the JSON file containing the label to category mapping")
    parser.add_argument("--use_gpu", action='store_true', default=False,
                        help="flag for indicating whether or not to use a GPU")
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['structure'],checkpoint['hidden_units'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state'])
    
    print("load_checkpoint-Done") 
    return model


def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((0, 0, 224, 224))

    np_image = np.array(img)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    
    
    return torch.FloatTensor(np_image)






def predict(image, model,use_gpu, topk=5):
    if use_gpu and torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
		
    model.to(device)    
    model.eval()
    image = image.unsqueeze_(0)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)   
    ps = torch.exp(output) 
        
    top_k, top_classes_idx = ps.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])
    
    
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    
    top_classes = []
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index])
    
    return list(top_k), list(top_classes)




def main():
    print("hi")
    args=get_args()
    cat_to_name = load_cat_names(args.category_names)
    model=load_checkpoint(args.checkpoint_vgg)
    print("\n Done building the model")
    processed_img=process_image(args.input_image)
    print("\n Done preprocess the image")
    top_p,top_classes=predict(processed_img, model, args.use_gpu,args.top_k)
    print("\n Done prediction!")
    labels = [cat_to_name[str(i)] for i in top_classes]
    print(f"Top {args.top_k} predictions are : {list(zip(labels, top_p))}")











if __name__ == '__main__':
    main()
