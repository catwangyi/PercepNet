import librosa
import torchaudio
import torch
import librosa.display
import matplotlib.pyplot as plt


def pitch_period_estimation(frame):
    pass

def pitch_coherence(spectrogram):
    pass

def ERB_Bands(spectrogram):
    torchaudio.functional.linear_fbanks()
    pass


if __name__ == "__main__":
    signal, sr = librosa.load('D:\PercepNet\\audio\p226_001.wav', sr=None)
    signal = torch.from_numpy(signal)
    spec_func = torchaudio.transforms.Spectrogram(n_fft=1024)
    signal_spectrogram = spec_func(signal)
    q = pitch_coherence(signal_spectrogram)
    # librosa.display.specshow(signal_spectrogram.numpy(), hop_length=512, y_axis='linear', x_axis='time', sr=sr)
    plt.show()
