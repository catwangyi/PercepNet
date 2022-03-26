import librosa
import numpy
import numpy as np
import torchaudio
import torch
from scipy.signal import find_peaks
import librosa.display
import matplotlib.pyplot as plt


def auto_correlation(signal1, signal2 = None):
    '''
    :param signal:
    :return:
    '''
    N = len(signal1)
    if signal2 is None:
        signal2 = signal1
    auto_corr = np.empty_like(signal1)
    for k in range(N):
        temp = 0
        for m in range(N - k - 1):
            temp += signal1[m] * signal2[m+k]
        auto_corr[k] = temp
    return auto_corr / auto_corr[0]


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


def pitch_period_estimation(signal, sr):
    '''
    :param signal: ndarray
    :return: pitch_period
    '''
    peak_index, _ = find_peaks(signal)
    pitch_period = peak_index[np.argmax(signal[peak_index])] / sr
    return pitch_period


def pitch_coherence(spectrogram):

    pass


def index_in_spectrogram(freq, spectrogram, sr):
    '''
    :param freq:
    :param spectrogram: N / 2
    :param sr:
    :return:
    '''
    # freq_bin
    freq_bin = sr / (len(spectrogram)*2)
    index = np.floor(freq / freq_bin)
    return index


def ERB_Bands(spectrogram):
    upper_freq = 20000
    lower_freq = 0
    band_num = 34
    erb_up = freq_2_erb(upper_freq)
    erb_low = freq_2_erb(lower_freq)
    erb_bands = np.linspace(start=erb_low, stop=erb_up, num=band_num)
    freq_bands = erb_2_freq(erb_bands)
    index_in_spectrogram()
    a = 1


def erb_2_freq(erb):
    return (np.power(10, erb/21.366)-1)/0.004368


def freq_2_erb(freq):
    return 21.366 * np.log10(0.004368*freq+1)


if __name__ == "__main__":
    pass
    # signal, sr = librosa.load("E:\pythonfiles\PercepNet\\audio\p226_001.wav", sr=None)
    # # x = np.ones(100)
    # # for i in range(len(x)):
    # #     x[i] = i/50
    # # signal = np.sin(2*np.pi*x)
    # signal = signal[:1024]
    # corr = auto_correlation(center_clipping(signal))
    #
    # pitch_period = pitch_period_estimation(corr, sr=sr)
    # print(pitch_period)
    #
    # plt.plot(signal)
    # plt.figure()
    # plt.plot(corr)
    # plt.show()
