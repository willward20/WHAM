import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv

class AutopilotDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=transforms.ToTensor()):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path) / 255.
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        ang_axis_val = self.img_labels.iloc[idx, 1].astype(np.float32)
        vel_axis_val = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image.float(), ang_axis_val, vel_axis_val


labels_path = "/home/pbd0/playground/wham_buggy/train/data/20230224_1803/labels.csv"
image_dir = "/home/pbd0/playground/wham_buggy/train/data/20230224_1803/images/"
dataset = AutopilotDataset(labels_path, image_dir)
print("data length: ", len(dataset))
# Define the size for train and test data
dataset_size = len(dataset)
train_size = round(dataset_size*0.9)
test_size = dataset_size - train_size
print(f"training dataset size: {train_size}, test dataset size: {test_size}")
# Load the datset (split into train and test)
train_data, test_data = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)
# image_sample, speed_sample, angle_sample = next(iter(train_dataloader))


class DonkeyNetwork(nn.Module):
    """
    Input image size: (120, 160, 3)
    """
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
        x = self.relu(self.conv24(x))  # (120-5)/2+1=58, (160-5)/2+1=78
        x = self.relu(self.conv32(x))  # (58-5)/2+1=27, (78-5)/2+1=37
        x = self.relu(self.conv64_5(x))  # (27-5)/2+1=12, (37-5)/2+1=17
        x = self.relu(self.conv64_3(x))  # 12-3+1=10, 17-3+1=15
        x = self.relu(self.conv64_3(x))  # 10-3+1=8, 15-3+1=13
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(120*160*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


############################################################################
#           M         M        A        IIIIIII     N        N             #
#           M M     M M       A A          I        N  N     N             # 
#           M  M   M  M      AAAAA         I        N    N   N             #
#           M    M    M     A     A        I        N      N N             # 
#           M         M    A       A    IIIIIII     N        N             #
############################################################################
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.
    for batch, (image, speed, angle) in enumerate(dataloader):
        target = torch.stack((speed, angle), -1)
        X, y = image.to(device), target.to(device)
        # Compute prediction error
        pred = model(X)
        batch_loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
        epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
        # if batch % 10 == 0:
        print(f"loss: {batch_loss:>7f} [{sample_count:>5d}/{size:>5d}]")

    return epoch_loss


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for image, speed, angle in dataloader:
            target = torch.stack((speed, angle), -1)
            X, y = image.to(device), target.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")

    return test_loss


# model = DenseNetwork().to(device)
model = DonkeyNetwork().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 16
train_losses, test_losses = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch_loss_train = train(train_dataloader, model, loss_fn, optimizer)
    train_losses.append(epoch_loss_train)
    epoch_loss_test = test(test_dataloader, model, loss_fn)
    test_losses.append(epoch_loss_test)
print("Done!")
plt.plot(list(range(epochs)), train_losses, '--', list(range(epochs)), test_losses)
plt.show()
#
# """
# # Load an image from the dataset and make a prediction
# image = read_image('images/200.jpg').to(DEVICE)  # read image to tensor
# image = (image.float() / 255 ) # convert to float and standardize between 0 and 1
# print("loaded image after divide and float: ", image.size())
# image = image.unsqueeze(dim=0) # add an extra dimension that is needed in order to make a prediction
# print("loaded image after unsqueeze: ", image.size())
# pred = model(image)
# print(pred)
# """
#
# Save the model
model_path = "/home/pbd0/playground/wham_buggy/train/models/donkey16epoch_20230224_1803.pth"
torch.save(model.state_dict(), model_path)
print(f"Saved autopilot model state to {model_path}")
