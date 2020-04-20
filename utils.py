import torch
torch.cuda.set_device(4)
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import *
#datasets & transformation
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


##training
def Train(epoch,train_loader,model,loss_type,optimizer):
    steps = len(train_loader)
    Loss=[]
    Acc =[]
    for i in range(epoch):
        total=0
        correct=0
        total_loss=0
        for j,(image,label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            loss = loss_type(output,label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,pred = torch.max(output.data,1)
            total += label.size(0)
            correct += (pred==label).sum().item()
            l=loss.item()
            total_loss += l

        print('Epoch:{}/{}, Training Loss:{:.2f}, Training Accuracy:{:.2f}'.format(i+1,epoch,total_loss/len(train_loader),100 * correct / total))

        Loss.append(total_loss/len(train_loader))
        Acc.append(100 * correct / total) 
    return Loss, Acc

        
        
    
#testing
def Test(test_loader,model,loss_type,optimizer):
    accuracy_test=[]
    model.eval()
    with torch.no_grad():
        total=0
        correct=0

        for j,(im_test,label_test) in enumerate(test_loader):
                im_test=im_test.cuda()
                label_test=label_test.cuda()
                output_test = model(im_test)
                loss_test = loss_type(output_test,label_test)


                _,pred = torch.max(output_test.data,1)
                total += label_test.size(0)
                correct += (pred==label_test).sum().item()
                acc_test = 100 * correct / total

                #print(correct)
        accuracy_test.append(acc_test)
        #accuracy_test_opt.append(acc_test)
        #accuracy_test_act.append(acc_test)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc_test))


#Actvation compare


def Visualize(loss,acc):
    fig= plt.figure()

    plt.plot([None]+loss, 'o-',label='Loss')
    plt.plot([None]+acc,'x-', label='Acc')
    plt.title('Training Loss & Accuracy of CNN')
    #plt.set_xlabel('Epochs')
    #bx.set_ylabel('Loss')
    #plt.legend(loc="upper right")
    plt.savefig('figure.png')
   

   