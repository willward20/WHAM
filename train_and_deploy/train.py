##################################################################
# Program Name: train.py
# Contributors: 
# 
# Train an autopilot for autonomous ground vehicle using
# convolutional neural network and labeled images. 
###################################################################


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
import matplotlib.pyplot as plt


# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


class NeuralNetwork(nn.Module):

    # Define CNN Architecture

    def __init__(self):
        super().__init__()

        # input image size: 300 x 300
        # input channel size: 3 (RGB)
        # conv layer output size = [(input_width - kernel + 2*padding) / stride] + 1
        # nn.Conv2d(input_channels, output_channels, kernel)

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


class CustomImageDataset(Dataset): 

    # Create a dataset from our collected data

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255 
        #print(image.float().size())
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image.float(), steering, throttle



def train(dataloader, model, loss_fn, optimizer):
    
    # Define Training Function
    
    num_batches = len(dataloader.dataset)
    model.train()
    
    train_loss = 0.0


    for batch, (X, steering, throttle) in enumerate(dataloader):
        #Combine steering and throttle into one tensor (2 columns, X rows)
        y = torch.stack((steering, throttle), -1) 
        #y = y.float()

        X, y = X.to(DEVICE), y.to(DEVICE)
        #print("Size X: ", X.size()) # torch.Size([BATCHSIZE, 3, 480, 640])

        # Compute prediction error
        pred = model(X)  # forward propagation
        loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        #if batch % 10 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"Train Loss: {loss:>7f}")
        train_loss += loss.item()
    #print("Average train loss: ", statistics.mean(train_loss))
    return train_loss/num_batches

        

def test(dataloader, model, loss_fn):
    
    # Define a test function to evaluate model performance

    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, steering, throttle in dataloader:

            #Combine steering and throttle into one tensor (2 columns, X rows)
            y = torch.stack((steering, throttle), -1) 
            y = y.float()
            
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y).item()
            #print(f"Test Loss: {loss:>7f}")
            test_loss += loss
            #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    #avg_loss = statistics.mean(test_loss)
    #print(f"Test Error: Avg loss: {avg_loss:>8f} \n")

    return test_loss/num_batches



def graph_data(x, train, test, TITLE, FILENAME):

    # Graph the test and train data

    fig = plt.figure()
    axs = fig.add_subplot(1,1,1)

    plt.plot(x, train, color='r', label="Training Loss")
    plt.plot(x, test, color='b', label='Testing Loss')
    axs.set_ylabel('Loss')
    axs.set_xlabel('Training Epoch')
    axs.set_title(TITLE)
    axs.legend()
    fig.savefig(FILENAME)

    return



if __name__ == '__main__':

    # Create a dataset
    annotations_file = "data2023-02-02-14-59/labels.csv"  # the name of the csv file
    img_dir = "data2023-02-02-14-59/images"  # the name of the folder with all the images in it
    collected_data = CustomImageDataset(annotations_file, img_dir)
    print("data length: ", len(collected_data))

    # Define the size for train and test data
    train_data_len = len(collected_data)
    train_data_size = round(train_data_len*0.9)
    test_data_size = round(train_data_len*0.1) 
    print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

    # Load the datset (split into train and test)
    train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
    train_dataloader = DataLoader(train_data, batch_size=50)
    test_dataloader = DataLoader(test_data, batch_size=50)
    epochs = 5



    # Initialize the model
    model = NeuralNetwork().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

    # Optimize the model
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loss = train(train_dataloader, model, loss_fn, optimizer)
        testing_loss = test(test_dataloader, model, loss_fn)
        print("average training loss: ", training_loss)
        print("average testing loss: ", testing_loss)
        # save values
        train_loss.append(training_loss)
        test_loss.append(testing_loss)   

    print(f"Optimize Done!")


    #print("final test lost: ", test_loss[-1])
    len_train_loss = len(train_loss)
    len_test_loss = len(test_loss)
    print("Train loss length: ", len_train_loss)
    print("Test loss length: ", len_test_loss)


    # create array for x values for plotting train
    epochs_array = list(range(1, epochs+1))
    print(epochs_array)

    graph_data(epochs_array, train_loss, test_loss, "TEST", "test.jpg")


    """
    # Load an image from the dataset and make a prediction
    image = read_image('images/200.jpg').to(DEVICE)  # read image to tensor
    image = (image.float() / 255 ) # convert to float and standardize between 0 and 1
    print("loaded image after divide and float: ", image.size())
    image = image.unsqueeze(dim=0) # add an extra dimension that is needed in order to make a prediction
    print("loaded image after unsqueeze: ", image.size())
    pred = model(image)
    print(pred)
    """

    # Save the model
    torch.save(model.state_dict(), "TEST.pth")
    print("Saved PyTorch Model State to TEST.pth")
