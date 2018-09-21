import os
import glob
import scipy.io.wavfile as wav
import numpy as np
from matplotlib import pyplot as plt
import librosa
import scipy


def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)

#merge_files_in_a_folder
def merge_files(path_read_folder, path_write_wav_file):
    #
    files = os.listdir(path_read_folder)
    merged_signal = []
    for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
        # print(filename)
        sr, signal = wav.read(filename)
        merged_signal.append(signal)
    merged_signal=np.hstack(merged_signal)
    merged_signal = np.asarray(merged_signal, dtype=np.int16)
    wav.write(path_write_wav_file, sr, merged_signal)

def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


def signal_by_db(x1, x2, snr, handle_method):
    from numpy.linalg import norm
    from math import sqrt
    import math

    #
    # target_clean_rms = 0.020
    # x1 = x1 - np.mean(x1)
    # stand_rms = np.sqrt(np.mean(x1))
    #
    # x1 = (target_clean_rms / stand_rms) * x1
    # stand_rms = np.sqrt(np.mean(x1))
    #
    # x2 = x2 - np.mean(x2)
    #
    # x2 = stand_rms / np.sqrt(np.mean(x2)) * x2

    x1 = x1.astype(np.int16)
    x2 = x2.astype(np.int16)
    x2= x2[:250000]
    l1 = x1.shape[0]
    l2 = x2.shape[0]



    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)

            if l2 < ll:
                x2_total = []
                for i in range(int(l1 // l2)*3):
                    x2_total.append(x2)
                x2_total = np.hstack(x2_total)
                #    x2 = np.append(x2, x2[:l2])

                ll2 = x1.shape[0]

                x2 = x2_total[:ll2]

    x2 = x2 / norm(x2) * norm(x1) / (10.0 ** ( 0.05*snr))
    # x2 = math.sqrt(np.sum(np.abs(x1) ** 2)) / math.sqrt((np.sum(np.abs(x2) ** 2)) * (10 ** snr)) * x2
    # x2 = np.sqrt(np.sum(np.abs(x1)**2))/np.sqrt((np.sum(np.abs(x2)**2))*(10**snr))*x2
    mix = x1 + x2

    return mix, x2


def generate_noise(x1, x2):
    x1 = x1.astype(np.int16)
    x2 = x2.astype(np.int16)
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    # print(x1.shape)
    # print(x2.shape)

    ll = max(l1, l2)
    x2_total=[]
    for i in range(int(l1 / l2) + 5):
        x2_total.append(x2)
    x2_total = np.hstack(x2_total)
    #    x2 = np.append(x2, x2[:l2])

    ll2 = x1.shape[0]

    x2 = x2_total[:ll2]

    return x2


def generate_noisy_data(path_clean_noise,nSNR):
    sr, clean_train = scipy.io.wavfile.read(path_clean_noise['path_clean_train_wav'])
    sr, clean_test = scipy.io.wavfile.read(path_clean_noise['path_clean_test_wav'])

    sr, noise_white = scipy.io.wavfile.read(path_clean_noise['path_noise_white_wav'])
    sr, noise_factory = scipy.io.wavfile.read(path_clean_noise['path_noise_factory_wav'])
    sr, noise_babble = scipy.io.wavfile.read(path_clean_noise['path_noise_babble_wav'])
    sr, noise_machinegun = scipy.io.wavfile.read(path_clean_noise['path_noise_machinegun_wav'])
    sr, noise_f16 = scipy.io.wavfile.read(path_clean_noise['path_noise_f16_wav'])
    sr, noise_buccaneer = scipy.io.wavfile.read(path_clean_noise['path_noise_buccaneer_wav'])

    for snr in nSNR:
        print("Gernerating Noisy SNR [%s] ................." % snr)

        # noisy_white_train, noise_white_total  = signal_by_db(clean_train, noise_white, snr, 'append')
        noisy_factory_train, noise_factory_total = signal_by_db(clean_train, noise_factory, snr, 'append')
        noisy_babble_train, noise_babble_total = signal_by_db(clean_train, noise_babble, snr, 'append')
        noisy_machinegun_train, noise_machinegun_total = signal_by_db(clean_train, noise_machinegun, snr, 'append')

        #matched noise
        noisy_white_test, _ = signal_by_db(clean_test, noise_white, snr, 'append')
        noisy_factory_test, _ = signal_by_db(clean_test, noise_factory, snr, 'append')
        noisy_babble_test, _ = signal_by_db(clean_test, noise_babble, snr, 'append')
        noisy_machinegun_test, _ = signal_by_db(clean_test, noise_machinegun, snr, 'append')
        #mismatched noise
        noisy_f16_test, _ = signal_by_db(clean_test, noise_f16, snr, 'append')
        noisy_buccaneer_test, _ = signal_by_db(clean_test, noise_buccaneer, snr, 'append')

        # noisy_white_train = np.asarray(noisy_white_train, dtype=np.int16)
        noisy_factory_train = np.asarray(noisy_factory_train, dtype=np.int16)
        noisy_babble_train = np.asarray(noisy_babble_train, dtype=np.int16)
        noisy_machinegun_train = np.asarray(noisy_machinegun_train, dtype=np.int16)

        # noisy_white_test = np.asarray(noisy_white_test, dtype=np.int16)
        noisy_factory_test = np.asarray(noisy_factory_test, dtype=np.int16)
        noisy_babble_test = np.asarray(noisy_babble_test, dtype=np.int16)
        noisy_machinegun_test = np.asarray(noisy_machinegun_test, dtype=np.int16)
        noisy_f16_test = np.asarray(noisy_f16_test, dtype=np.int16)
        noisy_buccaneer_test = np.asarray(noisy_buccaneer_test, dtype=np.int16)


        # noise_white_total = np.asarray(noise_white_total, dtype=np.int16)
        noise_factory_total = np.asarray(noise_factory_total, dtype=np.int16)
        noise_babble_total = np.asarray(noise_babble_total, dtype=np.int16)
        noise_machinegun_total = np.asarray(noise_machinegun_total, dtype=np.int16)


        PATH_ROOT = os.getcwd()
        makedirs(os.path.join(PATH_ROOT, "datasets/timit_noisy_selected/train/"))
        makedirs(os.path.join(PATH_ROOT, "datasets/timit_noisy_selected/test_match/"))
        makedirs(os.path.join(PATH_ROOT, "datasets/timit_noisy_selected/test_mismatch/"))
        makedirs(os.path.join(PATH_ROOT, "datasets/timit_noise_selected/"))

        # wav.write("datasets/timit_noisy/train/timit_noisy_white_snr%s_train.wav"%snr, sr, noisy_white_train)
        wav.write("datasets/timit_noisy_selected/train/timit_noisy_factory_snr%s_train.wav"%snr, sr, noisy_factory_train)
        wav.write("datasets/timit_noisy_selected/train/timit_noisy_babble_snr%s_train.wav"%snr, sr, noisy_babble_train)
        wav.write("datasets/timit_noisy_selected/train/timit_noisy_machinegun_snr%s_train.wav"%snr, sr, noisy_machinegun_train)

        #matched
        # wav.write("datasets/timit_noisy/test/timit_noisy_white_snr%s_test.wav"%snr, sr, noisy_white_test)
        wav.write("datasets/timit_noisy_selected/test_match/timit_noisy_factory_snr%s_test.wav"%snr, sr, noisy_factory_test)
        wav.write("datasets/timit_noisy_selected/test_match/timit_noisy_babble_snr%s_test.wav"%snr, sr, noisy_babble_test)
        wav.write("datasets/timit_noisy_selected/test_match/timit_noisy_machinegun_snr%s_test.wav"%snr, sr, noisy_machinegun_test)
        #mismatched
        wav.write("datasets/timit_noisy_selected/test_mismatch/timit_noisy_f16_snr%s_test.wav"%snr, sr, noisy_f16_test)
        wav.write("datasets/timit_noisy_selected/test_mismatch/timit_noisy_buccaneer_snr%s_test.wav"%snr, sr, noisy_buccaneer_test)

        # wav.write("datasets/timit_noisy/test/timit_noise_white_snr%s_total.wav"%snr, sr, noise_white_total)
        wav.write("datasets/timit_noise_selected/timit_noise_factory_snr%s_total.wav"%snr, sr, noise_factory_total)
        wav.write("datasets/timit_noise_selected/timit_noise_babble_snr%s_total.wav"%snr, sr, noise_babble_total)
        wav.write("datasets/timit_noise_selected/timit_noise_machinegun_snr%s_total.wav"%snr, sr, noise_machinegun_total)

        # noisy_f16_train = signal_by_db(clean_train, noise_f16, snr, 'append')
        # noisy_buccaneer_train = signal_by_db(clean_train, noise_buccaneer, snr, 'append')
        # noisy_f16_test = signal_by_db(clean_test, noise_f16, snr, 'append')
        # noisy_buccaneer_test = signal_by_db(clean_test, noise_buccaneer, snr, 'append')

        # noise_f16_total
        # noise_buccaneer_total

        # wav.write("datasets/timit_noisy_f16_snr%s_train.wav"%snr, sr, noisy_f16_train)
        # wav.write("datasets/timit_noisy_buccaneer_snr%s_train.wav"%snr, sr, noisy_buccaneer_train)
        # wav.write("datasets/timit_noisy_f16_snr%s_test.wav"%snr, sr, noisy_f16_test)
        # wav.write("datasets/timit_noisy_buccaneer_snr%s_test.wav"%snr, sr, noisy_buccaneer_test)

        del noisy_factory_train, noisy_babble_train, noisy_machinegun_train
        del noisy_factory_test, noisy_babble_test, noisy_machinegun_test
        del noise_factory_total, noise_babble_total, noise_machinegun_total

        # print("Finished generating the Noisy SNR [%s] ................." % snr)


if __name__ == '__main__':
    nSNR = [-5, 0, 5, 10, 15, 20]
    nNoise = 3


    nTrain=len(nSNR)*nNoise
    nTest =len(nSNR)*nNoise

    PATH_ROOT = os.getcwd()
    makedirs(os.path.join(PATH_ROOT, "datasets/timit_clean_selected/train/"))
    makedirs(os.path.join(PATH_ROOT, "datasets/timit_clean_selected/test/"))

    
    # makedirs("datasets/timit_clean/train/")
    # makedirs("datasets/timit_clean/test/")
    # path_read_train_folder    = "datasets/timit_8k_wav/train"
    # path_write_train_wav_file = "datasets/timit_clean/train/timit_clean_train.wav"
    # path_read_test_folder    = "datasets/timit_8k_wav/test"
    # path_write_test_wav_file = "datasets/timit_clean/test/timit_clean_test.wav"

    path_read_train_folder    = "datasets/timit_clean_selected/train"
    path_write_train_wav_file = "datasets/timit_clean_selected/timit_clean_selected_train.wav"
    path_read_test_folder    = "datasets/timit_clean_selected/test"
    path_write_test_wav_file = "datasets/timit_clean_selected/timit_clean_selected_test.wav"

    #noise data path
    #factory, babble, machinegun noises from NOISEX-92 DB
    path_noise_white = "datasets/noise/NOISEX/white.wav"
    path_noise_factory = "datasets/noise/NOISEX/factory1.wav"
    path_noise_babble = "datasets/noise/NOISEX/babble.wav"
    path_noise_machinegun = "datasets/noise/NOISEX/machinegun.wav"

    # buccaneer, f16 noises from NOISEX-92 DB and Cafeteria noise from ITU-T recommendation P.501
    # were used additionally for the test in mismatched condition
    path_noise_f16 = "datasets/noise/NOISEX/f16.wav"
    path_noise_buccaneer = "datasets/noise/NOISEX/buccaneer1.wav"


    # merge train, test waves to a wave file each
    merge_files(path_read_train_folder, path_write_train_wav_file)
    merge_files(path_read_test_folder,path_write_test_wav_file)

    path_clean_noise = {'path_clean_train_wav': path_write_train_wav_file,
            'path_clean_test_wav': path_write_test_wav_file,
            'path_noise_white_wav': path_noise_white,
            'path_noise_factory_wav': path_noise_factory,
            'path_noise_babble_wav': path_noise_babble,
            'path_noise_machinegun_wav': path_noise_machinegun,
            'path_noise_f16_wav': path_noise_f16,
            'path_noise_buccaneer_wav': path_noise_buccaneer
            }

    generate_noisy_data(path_clean_noise,nSNR)

    # make size of "clean_train" to size of "len(nSNR)*nNoise"
    sr, clean_train = scipy.io.wavfile.read(path_write_train_wav_file)
    clean_train_list=[]

    for i in range(int(nTrain)):
        clean_train_list.append(clean_train)
    clean_train_list=np.hstack(clean_train_list)

    sr, clean_test = scipy.io.wavfile.read(path_write_test_wav_file)
    clean_test_list=[]

    for i in range(int(nTest)):
        clean_test_list.append(clean_test)
    clean_test_list=np.hstack(clean_test_list)

    #clean total
    clean_train_list = np.asarray(clean_train_list, dtype=np.int16)
    wav.write("datasets/timit_clean_selected_train_total.wav", sr, clean_train_list)
    clean_test_list = np.asarray(clean_test_list, dtype=np.int16)
    wav.write("datasets/timit_clean_selected_test_total.wav", sr, clean_test_list)
    del clean_train_list, clean_test_list

    #noise total
    path_read_folder    = "datasets/timit_noise_selected/"
    path_write_wav_file = "datasets/timit_noise_selected_total.wav"
    merge_files(path_read_folder, path_write_wav_file)

    #noisy train total
    path_read_folder    = "datasets/timit_noisy_selected/train/"
    path_write_wav_file = "datasets/timit_noisy_selected_train_total.wav"
    merge_files(path_read_folder, path_write_wav_file)

    #matched noisy test total
    path_read_folder    = "datasets/timit_noisy/test_match/"
    path_write_wav_file = "datasets/timit_noisy_test_matched_total.wav"
    merge_files(path_read_folder, path_write_wav_file)

    #mismatched noisy test total
    path_read_folder    = "datasets/timit_noisy/test_mismatch/"
    path_write_wav_file = "datasets/timit_noisy_test_mismatched_total.wav"
    merge_files(path_read_folder, path_write_wav_file)