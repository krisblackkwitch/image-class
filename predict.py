import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import time
from torchvision import models
from torch.autograd import Variable
import seaborn as sns
import os
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,  help="defines the base value")
parser.add_argument('--image_path', type=str, help="defines the base value")




parser.add_argument('--top_k', type=int, default=5, help="defines the base value")
parser.add_argument('----category_names', type=str, default='cat_to_name.json', help="defines the base value")
parser.add_argument('--gpu', type=str, default='cuda', help="defines the base value")
args = parser.parse_args()
                    
                    
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

                 
                    

def load_checkpoint(checkpoint_path):
   
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc:storage)
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    for param in model.classifier.parameters():
        param.requires_grad= True
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = nn.Sequential(
                            nn.Linear(25088, 4096), 
                            nn.ReLU(), 
                            nn.Dropout(0.4),
                            nn.Linear(4096, 102),                   
                            nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model


model = load_checkpoint(args.checkpoint)
model









from PIL import Image

def process_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    image_precessed = transform(img)
    
    return image_precessed


image_path = args.image_path
img = Image.open(image_path)
img
img=process_image(image_path)
img.shape


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[26]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax




def predict(image_path, model, k, cat_to_name):
    
    
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_classes = probs.topk(k)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_classes[0].numpy()

    top_classes = []
    for classe in np_top_labs:
        top_classes.append(int(idx_to_class[classe]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_classes]
    
    return top_probs, top_classes, top_flowers
top_k=args.top_k
predict(image_path, model, top_k, cat_to_name)



def implementation(image_path, model, top_k, cat_to_name):
    # Sets up our plot
 
    # Set up title
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num] # Calls dictionary for name
    # Plot flower
    img = process_image(image_path)
  
    
    # Make prediction
    top_probs, top_classes, top_flowers = predict(image_path, model,top_k, cat_to_name) 
    top_probs = top_probs[0].detach().numpy() #converts from tensor to nparray
    # Plot bar chart

    

    print(top_probs, top_classes, top_flowers)


                    
implementation(image_path, model, top_k, cat_to_name)
# In[ ]:



