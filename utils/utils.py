import librosa
import numpy as np
import torchaudio
import torch
import librosa.display
import matplotlib.pyplot as plt


def auto_correlation(signal, clipped_signal = None):
    '''
    :param signal:
    :return:
    '''
    if clipped_signal is None:
        clipped_signal = signal
    auto_corr = np.empty_like(signal)
    for i in auto_corr:




    return auto_corr


def center_clipping(signal):
    '''
    :param signal:
    :return:
    '''
    # signal = np.array(((2, -1, 4), (2, -5, 3)))
    threshold = np.max(np.abs(signal)) * 0.68
    signal = np.where(np.abs(signal) <= threshold, 0, signal)
    signal = np.where(signal > threshold, 1, signal)
    # print(signal)
    signal = np.where(signal < -threshold, -1, signal)
    # print(signal)
    clipped_signal = signal
    return clipped_signal


def pitch_period_estimation(signal):
    pitch_period = signal
    '''
    :param signal: ndarray
    :return:
    '''
    return pitch_period

def pitch_coherence(spectrogram):
    pass

def ERB_Bands(spectrogram):
    torchaudio.functional.linear_fbanks()
    pass


if __name__ == "__main__":
    signal, sr = librosa.load('D:\PercepNet\\audio\p226_001.wav', sr=None)
    cliped_signal = center_clipping(signal[:1024])




    signal = torch.from_numpy(signal)
    spec_func = torchaudio.transforms.Spectrogram(n_fft=1024)
    signal_spectrogram = spec_func(signal)
    q = pitch_coherence(signal_spectrogram)
    # librosa.display.specshow(signal_spectrogram.numpy(), hop_length=512, y_axis='linear', x_axis='time', sr=sr)
    plt.show()
