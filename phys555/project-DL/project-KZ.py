from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class OrcaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        return

orca_dataset = OrcaDataset(csv_file='C:/Users/kzammit/Documents/PHYS555/orcas_classification/annotations-edited.csv',
                                    root_dir='C:/Users/kzammit/Documents/PHYS555/orcas_classification/cropped_spectrograms')




