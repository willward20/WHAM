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


class dense_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*300*300, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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
class DonkeyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(64*30*30, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160 images
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

