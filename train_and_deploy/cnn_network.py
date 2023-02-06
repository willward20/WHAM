##################################################################
# Program Name: cnn_network.py
# Contributors: 
# 
# Define CNN architectures used to train autopilot. 
###################################################################

import torch.nn as nn
import torch
import torch.nn.functional as F


################################################################################
# input image size: 300 x 300
# input channel size: 3 (RGB)
# conv layer output size = [(input_width - kernel + 2*padding) / stride] + 1
# nn.Conv2d(input_channels, output_channels, kernel)
#################################################################################


class cnn_network(nn.Module):

    # Define CNN Architecture

    def __init__(self):
        
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5) # (300 - 5 + 0) / 1 + 1 = 296 --> 6 x 296 x 296
        self.conv2 = nn.Conv2d(6, 12, 5) # (296 - 5) + 1 = 292 --> 12 x 292 x 292
        self.fc1 = nn.Linear(12*292*292, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x): # this defines the order that layers are executed
        x = F.relu(self.conv1(x)) # conv1 --> relu activation
        x = F.relu(self.conv2(x)) # conv2 --> relu activation
        x = torch.flatten(x, 1)   # flattens tensor, except the batch dimension
        x = F.relu(self.fc1(x))   # fc1 --> relu activation
        x = F.relu(self.fc2(x))   # fc2 --> relu activation
        x = self.fc3(x)           # fc3 --> two outputs (steering, throttle)
        return x


# Dokney docs --> parts --> keras and fastai (uses PyTorch)
# donkeycar (github) --> donkeycar --> parts --> keras.py and fastai.py (try fastai first)
# https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py
class donkey_net(nn.Module):
    def __init__(self):
        
        super(donkey_net, self).__init__()
        #self.dropout = 0.1

        # init the layers (kernel sizes were changed from 5 to 6 to avoid decimals)
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(6 , 6), stride=(2, 2)) # (300 - 6 + 2*0) / 2 + 1 = 148
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(6, 6), stride=(2, 2)) # (148 - 6)/2 + 1 = 72
        self.conv64_6 = nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(2, 2)) # (72 - 6)/2 + 1 = 34
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)) # (34 - 3)/1 + 1 = 32
        self.fc1 = nn.Linear(64*32*32, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2) # added this line from train.py
        #self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        #self.output1 = nn.Linear(50, 1)
        #self.output2 = nn.Linear(50, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))
        #x = self.drop(x)
        x = self.relu(self.conv32(x))
        #x = self.drop(x)
        x = self.relu(self.conv64_6(x))
        #x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        #x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        #x = self.drop(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        #x = self.drop(x)
        x = self.relu(self.fc2(x))
        #x1 = self.drop(x)
        x = self.fc3(x) # added this line
        return x
        
        #angle = self.output1(x1)
        #throttle = self.output2(x1)
        #return torch.cat((angle, throttle), 1)