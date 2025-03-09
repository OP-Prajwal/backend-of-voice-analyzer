import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



DATASET_PATH = os.path.abspath("../archive/KAGGLE/AUDIO")


# Function to plot waveform
def plot_waveform(file_path, title):
    y, sr = librosa.load(file_path, sr=22050)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {title}')
    plt.show()

# Function to plot spectrogram
def plot_spectrogram(file_path, title):
    y, sr = librosa.load(file_path, sr=22050)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {title}')
    plt.show()

# Function to plot MFCC

def plot_mfcc(file_path, title):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC - {title}')
    plt.show()

# Load and plot one Human and one AI audio
real_file = os.path.join(DATASET_PATH, "REAL", os.listdir(os.path.join(DATASET_PATH, "REAL"))[0])
fake_file = os.path.join(DATASET_PATH, "FAKE", os.listdir(os.path.join(DATASET_PATH, "FAKE"))[0])

# print("✅ Plotting Human Voice")
# plot_waveform(real_file, "Human Voice")
# plot_spectrogram(real_file, "Human Voice")
# plot_mfcc(real_file, "Human Voice")

print("✅ Plotting AI Voice")
plot_waveform(fake_file, "AI Voice")
plot_spectrogram(fake_file, "AI Voice")
plot_mfcc(fake_file, "AI Voice")

print("✅ All plots done!")
