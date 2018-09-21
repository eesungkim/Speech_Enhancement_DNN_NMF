# coding = utf-8
import numpy as np
from matplotlib import pyplot as plt
import librosa
import scipy


def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


def signal_by_db(x1, x2, snr, handle_method):
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)
            print(ll)
            if l1 < ll:
                x1 = np.append(x1, x1[:ll-l1])
            if l2 < ll:
                for i in range(int(l1/l2)+5):
                    x2 = np.append(x2, x2[:ll])

                ll2 = min(x1.shape[0], x2.shape[0])
                x1 = x1[:ll2]
                x2 = x2[:ll2]

    from numpy.linalg import norm
    x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr))
    mix = x1 + x2

    return mix


if __name__ == '__main__':
    num_FFT=512
    hop_size=128

    sr, speech_data =  scipy.io.wavfile.read(u"/datasets/timit/test/sa1.wav")
    sr, noise_data = scipy.io.wavfile.read('/datasets/noise/NOISEX/white.wav')
    plt.figure(figsize=(10, 10))

    S = librosa.stft(speech_data, n_fft=num_FFT, hop_length=hop_size, window='hanning')
    S=np.log(np.abs(S)**2)
    plt.subplot(311)
    plt.imshow(librosa.power_to_db(librosa.feature.melspectrogram(y=speech_data, sr=sr, n_fft=num_FFT, hop_length=hop_size),ref=np.max), cmap="hot")
    plt.title('Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    noisy_speech = signal_by_db(speech_data, noise_data, 15, 'cut')
    S = librosa.stft(speech_data, n_fft=num_FFT, hop_length=hop_size, window='hanning')
    S=np.log(np.abs(S)**2)
    plt.subplot(312)
    plt.imshow(librosa.power_to_db(librosa.feature.melspectrogram(y=noisy_speech, sr=sr, n_fft=num_FFT, hop_length=hop_size),ref=np.max), cmap="hot")
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    noisy_speech = signal_by_db(speech_data, noise_data, 0, 'cut')
    S = librosa.stft(speech_data, n_fft=num_FFT, hop_length=hop_size, window='hanning')
    S=np.log(np.abs(S)**2)
    plt.subplot(313)
    plt.imshow(librosa.power_to_db(librosa.feature.melspectrogram(y=noisy_speech, sr=sr, n_fft=num_FFT, hop_length=hop_size),ref=np.max), cmap="hot")
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.show()
