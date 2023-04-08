from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
import os.path, sys
import tensorflow as tf
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import wave
from IPython.display import Audio
from torchvision.io import read_image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
        #transforms.Normalize(mean=[0.485],
        #                     std=[0.229]), ])
    return transform


class OrcaImageDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

annotations_file = r"C:\Users\kzammit\Documents\PHYS555\orcas_classification\annotations-edited.csv"
img_dir = r"C:\Users\kzammit\Documents\PHYS555\orcas_classification\specs"
transforms_eff = preprocess()
orca_dataset = OrcaImageDataset(annotations_file, img_dir, transform=transforms_eff)

labels = orca_dataset.img_labels

xtrain_data, test_data, ytrain_labels, test_labels = train_test_split(
    orca_dataset, labels, test_size=0.2, random_state=21)

#  split a validation set from the training set
train_data, val_data, train_labels, val_labels = train_test_split(
    xtrain_data, ytrain_labels, test_size=0.2, random_state=22)

print('The number of samples for training is ' + str(len(train_data)))
print('The number of samples for validation is ' + str(len(val_data)))
print('The number of samples for testing is ' + str(len(test_data)))

scaler = StandardScaler()
#train_data_sc = scaler.fit_transform(train_data)
#val_data_sc = scaler.transform(val_data)
#test_data_sc = scaler.transform(test_data)

batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def train_test_model(model, train_loader, val_loader, n_epochs, optimizer):
    criterion = nn.MSELoss()
    optimizer = optimizer
    model = model.train()
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        train_batch_losses = []
        val_batch_losses = []
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.reshape(-1, 28 * 28)
            # print(img.shape)
            output = model(img)
            loss = criterion(output, img.data)
            # ************************ backward *************************
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.item())

        model = model.eval()
        for data in val_loader:
            output = model(img)
            loss = criterion(output, img.data)
            val_batch_losses.append(loss.item())

        train_loss = np.mean(train_batch_losses)
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ***************************** log ***************************
        if epoch % 10 == 0:
            print(f"epoch [{epoch + 1}/{n_epochs}], Train loss:{train_loss: .4f} Valid:{val_loss: .4f}")

    return model.eval()


# Creating a DeepAutoencoder class
class DeepAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiating the model and hyperparameters
model = DeepAutoencoder().to(device)
n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model = train_test_model(model, train_loader, val_loader, n_epochs, optimizer)

print('test')