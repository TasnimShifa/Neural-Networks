import torch
torch.cuda.set_device(4)
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import *
#datasets & transformation
import torchvision as tv
import torchvision.transforms as transforms


## create class for classifier with forward pass
class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        
        #defining layers
        #conv layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        #maxpooling
        self.maxpool1 = nn.MaxPool2d(kernel_size =2, stride = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size =2, stride = 2)
        
        #Fully connected layer
        self.fc1 = nn.Linear(12*4*4,100)
        self.fc2 = nn.Linear(100,10)
        
    def forward(self, x):
        x = self.maxpool1(f.relu(self.conv1(x)))
        x = self.maxpool2(f.relu(self.conv2(x)))
        
        x = f.relu(self.fc1(x.reshape(-1,12*4*4)))
        x = self.fc2(x)
        
        return x   