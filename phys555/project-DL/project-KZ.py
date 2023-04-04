from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os.path, sys
import torch
import tensorflow as tf
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import wave
from IPython.display import Audio

def plot_waveform(idx, waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)
    plt.savefig(r'C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\wavpngs\wav-' + str(idx) + '.png')
    plt.close()

def plot_spectrogram(idx, specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    #axs.set_ylim(0, 100)
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.savefig(r'C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\spectrograms\spec-' + str(idx) + '.png')
    plt.close()

def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)

def crop(path, dirs):
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            # last one does the height
            # left top right bottom
            imCrop = im.crop((80, 58, 475, 425)) #corrected
            imCrop.save(f + '-crop.png', "PNG", quality=300)


class OrcaDataset(Dataset):
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

if __name__ == "__main__":

    ## Crop spectrograms ##
    #path = r"C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\cropped_spectrograms"
    #dirs = os.listdir(path)
    #crop(path, dirs)
    ##
    ## Create spectrograms ##

    #for ii in range(593, 594):
    #    path = r'C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\audio'
    #    file = '\\' + str(ii) + '.wav'

    #    fullfile = path + file

    #    SAMPLE_SPEECH = fullfile

    #    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

    #   plot_waveform(idx=str(ii), waveform=SPEECH_WAVEFORM, sr=SAMPLE_RATE, title="Original waveform")
    #    Audio(SPEECH_WAVEFORM.numpy(), rate=SAMPLE_RATE)

    #   n_fft = 2000
    #    win_length = 2000
    #    hop_length = 200

        # Define transform
    #    spectrogram = T.Spectrogram(
    #       n_fft=n_fft,
    #        win_length=win_length,
     #       hop_length=hop_length,
     #       center=True,
     #       pad_mode="reflect",
     #       power=2.0,
     #   )
     #   spec = spectrogram(SPEECH_WAVEFORM)
     #   plot_spectrogram(idx=str(ii), specgram=spec[0], title="Orca Callies")

    print('test')

    orca_dataset = OrcaDataset(csv_file='C:/Users/kzammit/Documents/PHYS555/orcas_classification/annotations-edited.csv',
                                        root_dir='C:/Users/kzammit/Documents/PHYS555/orcas_classification/cropped_spectrograms')




