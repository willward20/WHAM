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
<<<<<<< HEAD
=======
    """
    Input image size: (120, 160, 3)
    """
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
<<<<<<< HEAD
        self.fc1 = nn.Linear(64*30*30, 128)
=======
        self.fc1 = nn.Linear(64*8*8, 128)
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
<<<<<<< HEAD
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30
=======
        x = self.relu(self.conv24(x))  # (120-5)/2+1 = 58
        x = self.relu(self.conv32(x))  # (58-5)/2+1 = 27
        x = self.relu(self.conv64_5(x))  # (27-5)/2+1 = 12
        x = self.relu(self.conv64_3(x))  # 12-3+1 = 10
        x = self.relu(self.conv64_3(x))  # 10-3+1 = 8
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

