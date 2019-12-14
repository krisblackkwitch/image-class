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


parser.add_argument('--data_dir', type=str, default='flowers', help="defines the base value")
parser.add_argument('--save_dir', type=str, default='classifiervgg16.ph', help="defines the base value")

parser.add_argument('--learning_rate', type=float, default=0.001, help="defines the base value")
parser.add_argument('--hidden_units', type=int, default=4096, help="defines the base value")
parser.add_argument('--epochs', type=int, default=3, help="defines the base value")
parser.add_argument('--gpu', type=str, default='cuda', help="defines the base value")
parser.add_argument('--arch', type=str, default='vgg13', help="defines the exponent value")
args = parser.parse_args()








data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(root=train_dir,  transform=data_transforms)
valid_set = datasets.ImageFolder(root=valid_dir,  transform=test_transforms)
test_set = datasets.ImageFolder(root=test_dir,  transform=test_transforms)


dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=50, shuffle=True,
                                      )
testloader = torch.utils.data.DataLoader(test_set, batch_size=50, 
                                       shuffle=False)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=50, shuffle=True)


trainiter = iter(dataloaders)
images, classes = next(trainiter)
print(images.shape)




arch = args.arch

model = models.vgg13(pretrained=True)


for param in model.parameters():
    param.requires_grad = True

hidden_units=args.hidden_units
    
model.classifier = nn.Sequential(
                      nn.Linear(25088, hidden_units), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(4096, 102),                   
                      nn.LogSoftmax(dim=1))






model.classifier





def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):

    model.train() # Puts model into training mode
    print_every = 40
    steps = 0
    use_gpu = False

    # Check to see whether GPU is available
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()
    else:
        model.cpu()

    # Iterates through each training pass based on #epochs & GPU/CPU
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

            # Forward and backward passes
            optimizer.zero_grad() # zero's out the gradient, otherwise will keep adding
            output = model.forward(inputs) # Forward propogation
            loss = criterion(output, labels) # Calculates loss
            loss.backward() # Calculates gradient
            optimizer.step() # Updates weights based on gradient & learning rate
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validation_loader)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accuracy: {:.3f}".format(accuracy))







def validate(model, criterion, data_loader):
    
    model.eval() # Puts model into validation mode
    accuracy = 0
    test_loss = 0

    for inputs, labels in iter(data_loader):
        if torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True) 
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)



epochs = args.epochs
learning_rate = args.learning_rate

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train(model, epochs, learning_rate, criterion, optimizer, dataloaders, validloader)




save_dir=args.save_dir


model.class_to_idx = image_datasets.class_to_idx
model.cpu()
torch.save({'arch': arch,
            'model_state_dict': model.state_dict(), # Holds all the weights and biases
            'class_to_idx': model.class_to_idx},
             save_dir)





