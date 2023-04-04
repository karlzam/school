
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
#%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the ECG data
data = raw_data[:, 0:-1]

n_features = data.shape[1]

# split full data into training and test set
xtrain_data, test_data, ytrain_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21)

#  split a validation set from the training set
train_data, val_data, train_labels, val_labels = train_test_split(
    xtrain_data, ytrain_labels, test_size=0.2, random_state=22)


print(train_data.shape, val_data.shape, test_data.shape)

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data.copy())
val_data = scaler.transform(val_data.copy()) # this is right, right? don't fit on the val data, only the train data, and then transform all
test_data = scaler.transform(test_data.copy())

train_labels = train_labels.astype(bool)
val_labels = val_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_val_data = val_data[val_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_val_data = val_data[~val_labels]
anomalous_test_data = test_data[~test_labels]

plt.grid()
plt.plot(np.arange(n_features), normal_train_data[0])
plt.title("Normal")
plt.show()

plt.grid()
plt.plot(np.arange(n_features), anomalous_train_data[0])
plt.title("Anomalous")
plt.show()

batch_size = 10

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class ECGAutoencoder(nn.Module):

    def __init__(self, n_features, z_dim):
        super(ECGAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 140),
            nn.ReLU(True),
            nn.Linear(140, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, z_dim),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 140),
            nn.ReLU(True),
            nn.Linear(140, n_features),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, train_loader, val_loader, n_epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)  # picked L1 loss b/c tabular data  right

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model = model.train()  # should this be outside the for loop? does it matter?

        train_batch_losses = []

        for data in train_loader:
            # my code block _
            optimizer.zero_grad()  # zero the grad

            output = model(data)  # get the output from the data
            loss = criterion(output, data)  # calculate the loss between the predicted output
            loss.backward()  # back prop the loss

            optimizer.step()  # step forward

            # my code block ^

            train_batch_losses.append(loss.item())

        val_batch_losses = []
        model = model.eval()
        with torch.no_grad():
            for data in val_loader:
                # my code block _

                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, data)
                loss.backward()

                optimizer.step()

                # my code block ^

                val_losses.append(loss.item())

        train_loss = np.mean(train_batch_losses)
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Losses Train = {train_loss:.4f} Valid = {val_loss:.4f}")

    # plot losses
    ax = plt.figure().gca()
    ax.plot(train_losses)
    ax.plot(val_losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'])
    plt.title('Loss monitoring')
    plt.show()

    # return trained model and loss arrays.
    return model.eval()

model = ECGAutoencoder(n_features, 8)
model = train_model(model, train_loader, val_loader)

torch.save(model, 'ecg_autoencoder.pt')