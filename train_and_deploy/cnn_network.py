import torch.nn as nn
import torch
import torch.nn.functional as F
class cnn_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,12,3)
        self.fc1 = nn.Linear(3*60*80, 120) # most recent downstairs test was 3*640*480
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class donkey_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(64*30*30, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

