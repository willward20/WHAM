import torch.nn as nn
import torch
import torch.nn.functional as F

image_size = 300

class cnn_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,12,3)
        self.fc1 = nn.Linear(3*image_size*image_size, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
