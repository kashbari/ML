## Convolutional Neural Network (PyTorch)

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch.optim as optim #For defining loss and optimization
## Define Convolutional Neural Network
## 

class Conv_Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # 5x5 conv 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5) # 5x5 conv 
        #self.pool = nn.MaxPool2d(k, k)  #kxk non-overlapping window that takes max

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120) 
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)


    def forward(self,t):
        x = F.relu(self.conv1(t))    #Convolution->ReLU
        x = F.relu(self.conv2(t))    #Convolution->ReLU
        #x = self.pool(F.relu(self.conv2(t))) #Conv->ReLU->Pooling
        #print(x.size())
        x = x.view(-1, 12*4*4)  #Convert all 2D arrays to 1D arrays
        x = F.relu(self.fc1(t))     #New linear layer
        x = F.relu(self.fc2(t))     #New linear layer
        x = self.out(t)  
        return t

## Driver Code
# cnn = Conv_Neural_Network()
