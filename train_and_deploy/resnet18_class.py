import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

image_size = 300


def load_resnet18(num_classes=2):
    # Load the pre-trained model (on ImageNet)
    model = models.resnet18(weights=True)

    # Don't allow model feature extraction layers to be modified
    for layer in model.parameters():
        layer.requires_grad = False

    # Change the classifier layer
    model.fc = nn.Linear(512, 2)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


class ResNet18(pl.LightningModule):
    def __init__(self, input_shape=(100, 3, image_size, image_size), output_size=2):
        # input shapes T*C*H*W. T batch size
        super().__init__()

        # Used by PyTorch Lightning to print an example model summary
        self.example_input_array = torch.rand(input_shape)

        # Metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.valid_mse = torchmetrics.MeanSquaredError()

        self.model = load_resnet18(num_classes=output_size)

        self.inference_transform = self.get_default_transform

        # Keep track of the loss history. This is useful for writing tests
        self.loss_history = []

    def forward(self, x):
        # Forward defines the prediction/inference action
        return self.model(x)

    def get_default_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_items = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        return transforms.Compose(transform_items)
