# Load an image from the dataset and make a prediction
from torchvision.io import read_image
from resnet18_class import ResNet18

image_size = 300
input_shape = (100, 10, image_size, image_size)
model = ResNet18(input_shape=input_shape).to(device)
image = read_image('./data2023-01-27-14-17/images/570.jpg').to(device)  # read image to tensor
image = (image.float() / 255)  # convert to float and standardize between 0 and 1
print("loaded image after divide and float: ", image.size())
image = image.unsqueeze(dim=0)  # add an extra dimension that is needed in order to make a prediction
print("loaded image after unsqueeze: ", image.size())
pred = model(image)
print(pred)
