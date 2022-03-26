import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


if __name__ == "__main__":
    signal = np.random.randn(1000)
    fft = np.fft.ifft(signal)
    plt.figure()
    plt.plot(signal)
    plt.figure()
    plt.plot(fft)
    plt.show()

