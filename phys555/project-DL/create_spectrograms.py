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


for ii in range(593, 594):
    path = r'C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\audio'
    file = '\\' + str(ii) + '.wav'

    fullfile = path + file

    SAMPLE_SPEECH = fullfile

    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

    plot_waveform(idx=str(ii), waveform=SPEECH_WAVEFORM, sr=SAMPLE_RATE, title="Original waveform")
    Audio(SPEECH_WAVEFORM.numpy(), rate=SAMPLE_RATE)

    n_fft = 2000
    win_length = 2000
    hop_length = 200

    # Define transform
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    spec = spectrogram(SPEECH_WAVEFORM)

    plot_spectrogram(idx=str(ii), specgram=spec[0], title="Orca Callies")

# # times between which to extract the wave from
# start = 47 # seconds
# end = 48.5 # seconds
#
# # file to extract the snippet from
# with wave.open(r'C:\Users\kzammit\Repos\school\phys555\project-DL\1208795168.700521004318.wav', "rb") as infile:
#     # get file data
#     nchannels = infile.getnchannels()
#     sampwidth = infile.getsampwidth()
#     framerate = infile.getframerate()
#     # set position in wave to start of segment
#     infile.setpos(int(start * framerate))
#     # extract data
#     data = infile.readframes(int((end - start) * framerate))
#
# # write the extracted data to a new file
# with wave.open(r'C:\Users\kzammit\Repos\school\phys555\project-DL\my_out_file.wav', 'w') as outfile:
#     outfile.setnchannels(nchannels)
#     outfile.setsampwidth(sampwidth)
#     outfile.setframerate(framerate)
#     outfile.setnframes(int(len(data) / sampwidth))
#     outfile.writeframes(data)
#