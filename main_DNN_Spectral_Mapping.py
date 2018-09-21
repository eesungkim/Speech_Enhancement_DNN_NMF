# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:12:48 2018
@author: eesungkim
"""
import os
import librosa
import argparse
import numpy as np
import scipy.io.wavfile as wav
import librosa.display

from utils.utils import * 
from utils.estnoise_ms import * 
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
import keras
from keras import metrics
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.layers import LeakyReLU, PReLU, ELU
from numpy.linalg import norm


def DNN_Spectral_Mapping(args):
    """Speech Enhancement using DNN Spectral Mapping"""
    PATH_ROOT = os.getcwd()
    os.chdir(PATH_ROOT)

    # noisy_train ; input of DNN
    path_dnn_noisy_train                        = os.path.join(PATH_ROOT, args.input_noisy_train)
    dnn_magnitude_noisy_train,_,sr              = perform_stft(path_dnn_noisy_train, args)
    # dnn_magnitude_noisy_train= splice_frames(dnn_magnitude_noisy_train.T, args.left_context, args.right_context).T

    # clean_train ; output of DNN
    path_dnn_clean_train                        = os.path.join(PATH_ROOT, args.input_clean_train)
    dnn_magnitude_clean_train,_,_               = perform_stft(path_dnn_clean_train, args)

    # noise_train
    path_noise                                  = os.path.join(PATH_ROOT, args.input_noise)
    dnn_magnitude_noise_train,_,_               = perform_stft(path_noise, args)

    path_clean_test                             = os.path.join(PATH_ROOT , args.input_clean_test)
    (sr, clean_test)                            = wav.read(path_clean_test)

    # noisy_test
    path_noisy_test                             = os.path.join(PATH_ROOT, args.input_noisy_test)
    (sr, noisy_test) = wav.read(path_noisy_test)
    dnn_magnitude_noisy_test, dnn_phase_noisy_test, _   = perform_stft(path_noisy_test, args)
    # magnitude_noisy_test= splice_frames(magnitude_noisy_test.T, args.left_context, args.right_context).T

    X_train = np.log(dnn_magnitude_noisy_train.T**2)
    y_train = np.log(dnn_magnitude_clean_train.T**2)
    X_test  = np.log(dnn_magnitude_noisy_test.T**2)

    # DNN training stage
    #####################################################################################
    k.clear_session()
    def get_dnn_model(X_train, y_train, args):
        # LeakyReLU, PReLU, ELU, ThresholdedReLU, SReLU
        model = Sequential()
        model.add(Dense(args.n_hidden, input_dim=X_train.shape[1], init='glorot_normal'))  # glorot_normal,he_normal
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(args.drop_out))

        model.add(Dense(args.n_hidden, init='glorot_normal'))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(args.drop_out))

        model.add(Dense(args.n_hidden, init='glorot_normal'))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(args.drop_out))

        model.add(Dense(units=y_train.shape[1], init='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])
        # model.summary()
        return model

    model = get_dnn_model(X_train, y_train, args)
    with tf.device('/gpu:0'):
        model_info = model.fit(X_train, y_train, batch_size=args.n_batch, epochs=args.n_epoch)
    # plot_model_history(model_info)
    print("Training complete.")


    # Enhancement stage
    #####################################################################################
    magnitude_estimated_clean = model.predict(X_test).T
    magnitude_estimated_clean = np.exp(np.sqrt(magnitude_estimated_clean))
    # magnitude_estimated_clean = magnitude_estimated_clean.astype('int16')


    # magnitude_estimated_clean=norm(magnitude_estimated_clean)
    #Reconstruction
    stft_reconstructed_clean   = merge_magphase(magnitude_estimated_clean, dnn_phase_noisy_test)
    signal_reconstructed_clean =librosa.istft(stft_reconstructed_clean, hop_length=args.hop_size, window=args.window)
    signal_reconstructed_clean = signal_reconstructed_clean.astype('int16')
    #####################################################################################
    output_path_estimated_noisy_test = os.path.join(PATH_ROOT, args.output_file)
    wav.write(output_path_estimated_noisy_test,sr,signal_reconstructed_clean)

    # Display signals, spectrograms
    show_signal(clean_test,noisy_test,signal_reconstructed_clean,sr)
    show_spectrogram(clean_test,noisy_test, signal_reconstructed_clean, sr, args.num_FFT,args.hop_size)
    # =============================================================================
    # PESQ
    # =============================================================================
    # PATH_MATLAB='"C:/Program Files/MATLAB/R2014a/bin/matlab.exe"'

    # PATH_MATLAB1 = os.path.join(PATH_ROOT , 'PESQ_MATLAB/execute_pesq.m')
    # from pymatbridge import Matlab
    # mlab = Matlab()
    # mlab = Matlab(executable=PATH_MATLAB)
    # mlab.start()

    # #PATH_MATLAB1 = os.path.join(PATH_ROOT , "PESQ_MATLAB","execute_pesq.m")
    # result_PESQ = mlab.run_func(PATH_MATLAB1, {'arg1': sr})
    # noisy_original_PESQ = result_PESQ['result'][0][0]
    # enhanced_PESQ = result_PESQ['result'][1][0]
    # mlab.stop()

    # snr=args.input_noisy_test
    # name=snr[53:-9]
    # print("[%s]\n Original: %.2f\n Spectral-Mapping\t: %.2f"%(name,noisy_original_PESQ,enhanced_PESQ))

def parse_args():
    parser = argparse.ArgumentParser(description='DNN-Spectral-Mapping Speech Enhancement')
    parser.add_argument('--datasets_dir',       type=str, default='datasets/')
    #for NMF
    parser.add_argument('--input_clean_train',  type=str, default='datasets/timit_clean_selected_train_total.wav')
    parser.add_argument('--input_noisy_train',   type=str, default='datasets/timit_noisy_selected_train_total.wav')
    parser.add_argument('--input_noise',        type=str, default='datasets/timit_noise_selected_total.wav')

    parser.add_argument('--input_clean_test',   type=str, default='datasets/timit_clean_selected/timit_clean_selected_test.wav')
    parser.add_argument('--input_noisy_test',   type=str, default='datasets/timit_noisy_selected/test_match/timit_noisy_babble_snr10_test.wav')

    #for DNN
    parser.add_argument('--input_dnn_clean_train', type=str, default='datasets/timit_clean_train_total.wav')
    parser.add_argument('--input_dnn_noisy_train', type=str, default='datasets/timit_noisy_train_total.wav')
    parser.add_argument('--input_dnn_noise',    type=str, default='datasets/timit_noise_total.wav')
    parser.add_argument('--output_file',        type=str, default='datasets/output/estimated_clean_DNN-Spectral-Mapping.wav')

    parser.add_argument('--left-context',       type=int, dest="left_context", default=3, help="left context of inputs for neural networks")
    parser.add_argument('--right-context',      type=int, dest="right_context", default=3, help="right context of inputs for neural networks")

    parser.add_argument('--num_FFT',            type=int, default='512',    help='')
    parser.add_argument('--hop_size',           type=int, default='128',    help='')
    parser.add_argument('--window',             type=str, default='hamming',help='boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser')
    parser.add_argument('--n_epoch',    type=int, default='20', help='number of DNN epoch')
    parser.add_argument('--n_hidden',   type=int, default='1024', help='hidden units of DNN')
    parser.add_argument('--drop_out',   type=float, default='0.5', help='dropout of DNN')
    parser.add_argument('--n_batch',    type=int, default='128', help='mini batch size')

    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.datasets_dir):
        os.makedirs(args.datasets_dir)
    assert args.num_FFT >= 1, 'number of FFT size must be larger than or equal to one'
    assert args.hop_size < args.num_FFT, 'hop size must be smaller than number of FFT size'
    return args

if __name__ == '__main__':
    args = parse_args()
    DNN_Spectral_Mapping(args)
