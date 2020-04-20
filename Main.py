import argparse
import torch
torch.cuda.set_device(4)
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import *
#datasets & transformation
import torchvision as tv
import torchvision.transforms as transforms
from Model import CNN
from utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training on SST dataset using RNN')

    # model hyper-parameter variables
    parser.add_argument('--epoch', default=5, metavar='epoch', type=int, help='Number of Epochs')
    parser.add_argument('--lr', default=0.01, metavar='lr', type=float, help='Learning Rate')
    
     
    args = parser.parse_args()
    
    
    epoch = args.epoch
    lr = args.lr


    ## load dataset & dataloader
    train = tv.datasets.MNIST(root='./data/', train = True, transform = transforms.ToTensor(), download = True)
    test = tv.datasets.MNIST(root='./data',train=False, transform = transforms.ToTensor(), download = True)
    train_loader = torch.utils.data.DataLoader(train,100,True)
    test_loader = torch.utils.data.DataLoader(train,100,False)


    ## model object
    model = CNN()
    model.cuda()
    print(model)

    ## loss & optimizer
    loss_type = nn.CrossEntropyLoss()
    loss_type.cuda()
    optimizer = SGD(model.parameters(),lr) #0.005, 0.01

    Loss,Acc = Train(epoch,train_loader,model,loss_type,optimizer)
    Test(test_loader,model,loss_type,optimizer)
    Visualize(Loss, Acc)








