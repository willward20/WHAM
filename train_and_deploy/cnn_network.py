import torch.nn as nn
import torch
import torch.nn.functional as F

#image_size = 120*160

class cnn_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # (300 - 5 + 0) / 1 + 1 = 296 --> 6 x 296 x 296
        self.conv2 = nn.Conv2d(6, 12, 5)  # (296 - 5) + 1 = 292 --> 12 x 292 x 292
        self.fc1 = nn.Linear(12 * 292 * 292, 120) # changed from 292,120 to 12,9
        self.fc2 = nn.Linear(120,84) #84
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):  # this defines the order that layers are executed
        x = F.relu(self.conv1(x))  # conv1 --> relu activation
        x = F.relu(self.conv2(x))  # conv2 --> relu activation
        x = torch.flatten(x, 1)  # flattens tensor, except the batch dimension
        x = F.relu(self.fc1(x))  # fc1 --> relu activation
        x = F.relu(self.fc2(x))  # fc2 --> relu activation
        x = self.fc3(x)  # fc3 --> two outputs (steering, throttle)
        return x

class Linear(nn.Module):
    def __init__(self):
        #image = 3*160*120 = 57,600
        super(Linear, self).__init__()
        self.dropout = 0.1
        # init the layers
        # 3, 2, 5*5
        # 2,
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        # * 5*5
        self.fc1 = nn.Linear(6656, 100)
        self.fc2 = nn.Linear(100, 50)
        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.output1 = nn.Linear(50, 1)
        self.output2 = nn.Linear(50, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))
        x = self.drop(x)
        x = self.relu(self.conv32(x))
        x = self.drop(x)
        x = self.relu(self.conv64_5(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1 = self.drop(x)
        angle = self.output1(x1)
        throttle = self.output2(x1)
        return torch.cat((angle, throttle), 1)

# donkeynet available on donkey car github 
class DonkeyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(64*8*13, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30 default 300*300
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    """
        for 120*160 images
        (120-5)/2+1 = 58
        (58-5)/2+1 = 27
        (27-5)/2+1 = 12
        12-3+1 = 10
        10-3+1 = 8

        height 
        (160-5)/2+1 = 78
        78-5)/2+1 = 37
        (37-5)/2+1 = 17
        17-3+1 = 15
        15-3+1 = 13
    """