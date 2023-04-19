
# Train an autopilot for autonomous ground vehicle using
# convolutional neural network and labeled images. 


import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import cnn_network
import cv2 as cv
from tqdm import tqdm


# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


class CustomImageDataset(Dataset): 

    # Create a dataset from our collected data

    def __init__(self, annotations_file, img_dir, transform=transforms.ToTensor()):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        steer = self.img_labels.iloc[idx, 1].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image.float(), steer



def train(dataloader, model, loss_fn, optimizer):
    num_samples = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0

    for batch, (image, steer) in enumerate(dataloader):
        # Combine steering and throttle into one tensor (2 columns, X rows)
        X, y = image.to(DEVICE), steer.to(DEVICE)

        # Compute prediction error
        pred = model(X)  # forward propagation
        batch_loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
        epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
        # print(f"loss: {batch_loss} [{sample_count}/{num_samples}]")

    return epoch_loss

        

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for image, steer in dataloader:
            #Combine steering and throttle into one tensor (2 columns, X rows)
            X, y = image.to(DEVICE), steer.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    # print(f"Test Error: {test_loss:>8f} \n")

    return test_loss


if __name__ == '__main__':

    # Create a dataset
    data_name = "2023_04_19_13_50-dummy"
    data_dir = os.path.join(sys.path[0], "data", data_name)
    annotations_file = f"{data_dir}/labels.csv"  # the name of the csv file
    img_dir = f"{data_dir}/images"  # the name of the folder with all the images in it
    collected_data = CustomImageDataset(annotations_file, img_dir)
    print("data length: ", len(collected_data))

    # Define the size for train and test data
    train_data_len = len(collected_data)
    train_data_size = round(train_data_len*0.95)
    test_data_size = train_data_len - train_data_size 
    print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

    # Load the datset (split into train and test)
    train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
    train_dataloader = DataLoader(train_data, batch_size=256)
    test_dataloader = DataLoader(test_data, batch_size=256)

    # Initialize the model
    model_dir = os.path.join(sys.path[0], "models", data_name)
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    epochs = 30
    learning_rate = 0.001
    title=f"DonkeyNet-epochs_{epochs}-lr_{learning_rate}"
    model = cnn_network.DonkeyNet().to(DEVICE) # choose the architecture class from cnn_network.py
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    # Optimize the model
    train_losses = []
    test_losses = []
    pbar = tqdm(range(epochs))
    for t in pbar:
        pbar.set_description('epochs {}'.format(t + 1))
        try:
            training_loss = train(train_dataloader, model, loss_fn, optimizer)
            testing_loss = test(test_dataloader, model, loss_fn)
            print("average training loss: ", training_loss)
            print("average testing loss: ", testing_loss)
            # save losses
            train_losses.append(training_loss)
            test_losses.append(testing_loss)
            # Save model every 5 epochs
            if t >= 9 and not (t+1) % 5: 
                torch.save(model.state_dict(), model_dir+f"/{title}_{t+1}.pth")
                print(f"\nSaved PyTorch Model State to {model_dir}/{title}_{t+1}.pth")

        except Exception as e:
            print(e)

    print(f"Optimize Done!")

    # Graph the test and train data
    plt.plot(list(range(epochs)), train_losses, '--', list(range(epochs)), test_losses)
    plt.savefig(os.path.join(model_dir, title+".png"))

