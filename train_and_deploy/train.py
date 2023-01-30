"""
Template for training data with a NN model.
"""
import os
import statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
import matplotlib.pyplot as plt
from cnn_network import cnn_network
from resnet18_class import Resnet18

DEVICE = torch.device("cuda")

#############################################
# Class Definitions
#############################################

# Class for creating a dataset from our collected data
class CustomImageDataset(Dataset):
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


###############################################
# Function definitions
###############################################

# Define Training Function
def train(dataloader, model, loss_fn, optimizer):
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

        
# Define a test function to evaluate model performance
def test(dataloader, model, loss_fn):
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


# Graph the test and train data
def graph_data(x, train, test, TITLE, FILENAME):

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



############################################################################
#           M         M        A        IIIIIII     N        N             #
#           M M     M M       A A          I        N  N     N             # 
#           M  M   M  M      AAAAA         I        N    N   N             #
#           M    M    M     A     A        I        N      N N             # 
#           M         M    A       A    IIIIIII     N        N             #
############################################################################


# Create a dataset
annotations_file = "labels_edited.csv"  # the name of the csv file
img_dir = "images_edited"  # the name of the folder with all the images in it
collected_data = CustomImageDataset(annotations_file, img_dir)
print("data length: ", len(collected_data))

# Define the size for train and test data
train_data_len = len(collected_data)
train_data_size = round(train_data_len*0.9)
test_data_size = round(train_data_len*0.1) 
print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

# Load the datset (split into train and test)
train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
train_dataloader = DataLoader(train_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)
epochs = 5

# Initialize the model
input_shape = (100, 3, image_size,image_size)
model = ResNet18(input_shape=input_shape).to(DEVICE)

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

graph_data(epochs_array, train_loss, test_loss, "Resnet18", "resnet18_01-29.jpg")


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
torch.save(model.state_dict(), "resnet18_01-29.pth")
print("Saved PyTorch Model State to resnet18_01-29.pth")
