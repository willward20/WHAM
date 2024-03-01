
# Train an autopilot for autonomous ground vehicle using
# convolutional neural network and labeled images. 


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
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image.float(), steering, throttle



def train(dataloader, model, loss_fn, optimizer):
    
    # Define Training Function
    
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0

    for batch, (image, steering, throttle) in enumerate(dataloader):
        # Combine steering and throttle into one tensor (2 columns, X rows)
        target = torch.stack((steering, throttle), -1) 
        X, y = image.to(DEVICE), target.to(DEVICE)

        # Compute prediction error
        pred = model(X)  # forward propagation
        batch_loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
        epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
        
    return epoch_loss

        

def test(dataloader, model, loss_fn):
    
    # Define a test function to evaluate model performance

    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for image, steering, throttle in dataloader:
            #Combine steering and throttle into one tensor (2 columns, X rows)
            target = torch.stack((steering, throttle), -1) 
            X, y = image.to(DEVICE), target.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches

    return test_loss

# Graph the test and train data
def graph_data(x, train, test, TITLE, FILENAME):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)

    plt.plot(x, train, color='b', label="Training Loss")
    plt.plot(x, test, color='r', label='Testing Loss')
    axs.set_ylabel('Loss')
    axs.set_xlabel('Training Epoch')
    axs.set_title(TITLE)
    axs.legend()
    fig.savefig(FILENAME)

    return


if __name__ == '__main__':

    # Create a dataset
    dir ="FOLDER" 
    annotations_file = f"{dir}/labels.csv"  # the name of the csv file
    img_dir = f"{dir}/images"  # the name of the folder with all the images in it
    collected_data = CustomImageDataset(annotations_file, img_dir)
    print("data length: ", len(collected_data))

    # Define the size for train and test data
    train_data_len = len(collected_data)
    train_data_size = round(train_data_len*0.9)
    test_data_size = train_data_len - train_data_size 
    print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

    # Load the datset (split into train and test)
    train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
    train_dataloader = DataLoader(train_data, batch_size=125)
    test_dataloader = DataLoader(test_data, batch_size=125)


    # Initialize the model
    # Models that train well:
    #     lr = 0.001, epochs = 10
    #     lr = 0.0001, epochs = 15 (epochs = 20 might also work)
    model = cnn_network.DonkeyNet().to(DEVICE) # choose the architecture class from cnn_network.py
    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    epochs = 2

    # Optimize the model
    train_loss = []
    test_loss = []
    folder_name= dir.split("/")[1]
    Title=f"{folder_name}_DonkeyNet_{epochs}_epochs_lr_{learning_rate}"
    pbar = tqdm(range(epochs))
    for t in pbar:
        pbar.set_description('epochs {}'.format(t + 1))
        try:
            training_loss = train(train_dataloader, model, loss_fn, optimizer)
            testing_loss = test(test_dataloader, model, loss_fn)
            print("average training loss: ", training_loss)
            print("average testing loss: ", testing_loss)
            # save values
            train_loss.append(training_loss)
            test_loss.append(testing_loss)

        except Exception as e:
            print(e)  

    print(f"Optimize Done!")


    #print("final test lost: ", test_loss[-1])
    len_train_loss = len(train_loss)
    len_test_loss = len(test_loss)
    print("\nTrain loss length: ", len_train_loss)
    print("\nTest loss length: ", len_test_loss)


    # create array for x values for plotting train
    epochs_array = list(range(epochs))

    # Graph the test and train data
    graph_data(epochs_array, train_loss, test_loss,Title, f"models/{Title}.jpg")

    # Save the model
    torch.save(model.state_dict(), f"models/{Title}.pth")
    print(f"\nSaved PyTorch Model State to models/{Title}.pth")


    
