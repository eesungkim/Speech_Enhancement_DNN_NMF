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
import tflearn
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


def NMF_DNN(args):
    """Speech Enhancement using NMF-DNN
    """
    PATH_MATLAB='"C:/Program Files/MATLAB/R2014a/bin/matlab.exe"'
    PATH_ROOT = os.getcwd() 
    PATH_MATLAB1 = os.path.join(PATH_ROOT , 'PESQ_MATLAB/execute_pesq.m')
    
    os.chdir(PATH_ROOT)

    #for NMF
    path_clean_train            = os.path.join(PATH_ROOT, args.input_clean_train)
    (sr, clean_train)           = wav.read(path_clean_train)
    stft_clean_train            = librosa.stft(clean_train, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    # stft_clean_train            = stft_clean_train[:,:10000]
    magnitude_clean_train, _    = divide_magphase(stft_clean_train, power=1)

    #for DNN
    path_dnn_clean_train        = os.path.join(PATH_ROOT , args.input_dnn_clean_train)
    (sr, dnn_clean_train)       = wav.read(path_dnn_clean_train)
    dnn_stft_clean_train        = librosa.stft(dnn_clean_train, n_fft=args.num_FFT, hop_length=args.hop_size,window=args.window)
    dnn_magnitude_clean_train, _= divide_magphase(dnn_stft_clean_train, power=1)


    path_dnn_noisy_train        = os.path.join(PATH_ROOT , args.input_dnn_noisy_train)
    (sr, dnn_noisy_train)       = wav.read(path_dnn_noisy_train)
    dnn_stft_noisy_train        = librosa.stft(dnn_noisy_train, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    dnn_magnitude_noisy_train, _= divide_magphase(dnn_stft_noisy_train, power=1)

    path_noise                  = os.path.join(PATH_ROOT, args.input_noise)
    (sr, noise_dnn)             = wav.read(path_noise)
    dnn_stft_noise_train        = librosa.stft(noise_dnn, n_fft=args.num_FFT, hop_length=args.hop_size,window=args.window)
    dnn_magnitude_noise_train, _= divide_magphase(dnn_stft_noise_train, power=1)

    #for Noise
    path_noise_1                = os.path.join(PATH_ROOT, args.input_noise_1)
    (sr, noise_1)               = wav.read(path_noise_1)
    stft_noise_1                = librosa.stft(noise_1, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    magnitude_noise_1, _        = divide_magphase(stft_noise_1, power=1)


    path_noise_2                = os.path.join(PATH_ROOT, args.input_noise_2)
    (sr, noise_2)               = wav.read(path_noise_2)
    stft_noise_2                = librosa.stft(noise_2, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)

    magnitude_noise_2, _        = divide_magphase(stft_noise_2, power=1)
    path_noise_3                = os.path.join(PATH_ROOT, args.input_noise_3)
    (sr, noise_3)               = wav.read(path_noise_3)
    stft_noise_3                = librosa.stft(noise_3, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    magnitude_noise_3, _        = divide_magphase(stft_noise_3, power=1)


    path_clean_test = os.path.join(PATH_ROOT, args.input_clean_test)
    (sr, clean_test) = wav.read(path_clean_test)


    path_noisy_test = os.path.join(PATH_ROOT, args.input_noisy_test)
    (sr, noisy_test) = wav.read(path_noisy_test)
    stft_noisy_test = librosa.stft(noisy_test, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    magnitude_noisy_test, phase_noisy_test = divide_magphase(stft_noisy_test, power=1)





    # NMF training stage
    #####################################################################################
    #obtain the basis matrix of clean_speech
    W_clean_train, H_clean_train  = NMF_MuR(magnitude_clean_train,args.r,args.max_iter,args.display_step,const_W=False,init_W=0)

    # noise
    ##########################################################
    # 1) 각 노이즈 마다 3000 frame 씩 이어붙혀서 총 9000으로 만들어서 40 base 로 만들기
    magnitude_noise_1, magnitude_noise_2, magnitude_noise_3= magnitude_noise_1[:,:3000],magnitude_noise_2[:,:3000],magnitude_noise_3[:,:3000]
    nmf_magnitude_noise = np.concatenate((magnitude_noise_1, magnitude_noise_2, magnitude_noise_3),axis=1)
    #obtain the basis matrix of noise
    W_noise, H_noise              = NMF_MuR(nmf_magnitude_noise,args.r,args.max_iter,args.display_step,const_W=False,init_W=0)


    # # 2) base 13,13,14로 이어 붙히기
    # # magnitude_noise_1, magnitude_noise_2, magnitude_noise_3 = magnitude_noise_1[:, :3000], magnitude_noise_2[:,:3000], magnitude_noise_3[:,:3000]
    #
    # #obtain the basis matrix of noise
    # W_noise_1, _ = NMF_MuR(magnitude_noise_1,13,args.max_iter,args.display_step,const_W=False,init_W=0)
    # W_noise_2, _ = NMF_MuR(magnitude_noise_2, 13, args.max_iter, args.display_step, const_W=False, init_W=0)
    # W_noise_3, _ = NMF_MuR(magnitude_noise_3, 14, args.max_iter, args.display_step, const_W=False, init_W=0)
    # ####################################################
    # W_noise = np.concatenate((W_noise_1, W_noise_2, W_noise_3),axis=1)

    # iH is the output of DNN
    _, H_NMF_encoding_estimated_clean = NMF_MuR(dnn_magnitude_clean_train,args.r,args.max_iter,args.display_step,const_W=True, init_W=W_clean_train)
    _, H_NMF_encoding_estimated_noise = NMF_MuR(dnn_magnitude_noise_train,args.r,args.max_iter,args.display_step,const_W=True, init_W=W_noise)


    # DNN training stage
    #####################################################################################
    X_train = dnn_magnitude_noisy_train.T
    y_train = np.concatenate([H_NMF_encoding_estimated_clean,H_NMF_encoding_estimated_noise], axis=0).T
    X_test = magnitude_noisy_test
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
        model_info = model.fit(X_train, y_train,
                               batch_size       = args.n_batch,
                               epochs           = args.n_epoch,
                               validation_split = 0.1)
    # plot_model_history(model_info)
    print("Training complete.")

    #Enhancement stage
    #####################################################################################
    H_DNN_for_NMF_encoding_matrix = model.predict(X_test.T).T

    H_estimated_from_DNN_clean = H_DNN_for_NMF_encoding_matrix[:args.r,:]
    H_estimated_from_DNN_noise = H_DNN_for_NMF_encoding_matrix[args.r:,:]

    magnitude_estimated_from_DNN_clean = np.matmul(W_clean_train,H_estimated_from_DNN_clean)
    magnitude_estimated_from_DNN_noise = np.matmul(W_noise,H_estimated_from_DNN_noise)

    #Gain function similar to wiener filter to enhance the speech signal
    wiener_gain = np.power(magnitude_estimated_from_DNN_clean,args.p) / \
                        ( np.power(magnitude_estimated_from_DNN_clean,args.p) + np.power(magnitude_estimated_from_DNN_noise, args.p))
    magnitude_estimated_clean = wiener_gain * magnitude_noisy_test

    #Reconstruction
    stft_reconstructed_clean = merge_magphase(magnitude_estimated_clean, phase_noisy_test)
    signal_reconstructed_clean =librosa.istft(stft_reconstructed_clean, hop_length=args.hop_size, window=args.window)
    signal_reconstructed_clean = signal_reconstructed_clean.astype('int16')
    #####################################################################################
    output_path_estimated_noisy_test = os.path.join(PATH_ROOT, args.output_file)
    wav.write(output_path_estimated_noisy_test,sr,signal_reconstructed_clean)

    # Display signals, spectrograms
    show_signal(clean_test,noisy_test,signal_reconstructed_clean,sr)
    show_spectrogram(clean_test,noisy_test, signal_reconstructed_clean, sr, args.frame_length,args.hop_size)
    # =============================================================================
    # PESQ
    # =============================================================================
    from pymatbridge import Matlab
    mlab = Matlab()
    mlab = Matlab(executable=PATH_MATLAB)
    mlab.start()
    #PATH_MATLAB1 = os.path.join(PATH_ROOT , "PESQ_MATLAB","execute_pesq.m")
    result_PESQ = mlab.run_func(PATH_MATLAB1, {'arg1': sr})
    noisy_original_PESQ = result_PESQ['result'][0][0]
    enhanced_PESQ = result_PESQ['result'][1][0]
    mlab.stop()
    snr=args.input_noisy_test
    name=snr[53:-9]
    print("[%s]\n Original: %.2f\n NMF-DNN\t: %.2f"%(name,noisy_original_PESQ,enhanced_PESQ))


    from line_notify import LineNotify
    notify = LineNotify("yIuEACAwFk8CaORZRmBtDNdZhdOCRB7SnMO1qIRf810")
    notify.send("[%s]\n Original: %.2f=\n NMF-DNN\t: %.2f"%(name,noisy_original_PESQ,enhanced_PESQ))


def parse_args():
    parser = argparse.ArgumentParser(description='NMF-DNN Speech Enhancement')
    parser.add_argument('--datasets_dir',       type=str, default='datasets/')
    #for NMF
    parser.add_argument('--input_clean_train',  type=str, default='datasets/timit_clean_selected_train_total.wav')
    parser.add_argument('--input_noisy_train',   type=str, default='datasets/timit_noisy_selected_train_total.wav')

    parser.add_argument('--input_clean_test',   type=str, default='datasets/timit_clean_selected/timit_clean_selected_test.wav')
    parser.add_argument('--input_noisy_test',   type=str, default='datasets/timit_noisy_selected/test_match/timit_noisy_babble_snr10_test.wav')

    #for noise
    parser.add_argument('--input_noise_1', type=str, default='datasets/noise/NOISEX/babble.wav')
    parser.add_argument('--input_noise_2', type=str, default='datasets/noise/NOISEX/factory1.wav')
    parser.add_argument('--input_noise_3', type=str, default='datasets/noise/NOISEX/machinegun.wav')

    #for DNN
    parser.add_argument('--input_dnn_clean_train', type=str, default='datasets/timit_clean_selected_train_total.wav')
    parser.add_argument('--input_dnn_noisy_train', type=str, default='datasets/timit_noisy_selected_train_total.wav')
    parser.add_argument('--input_noise',           type=str, default='datasets/timit_noise_selected_total.wav')

    parser.add_argument('--output_file',        type=str, default='datasets/output/estimated_clean_NMF-DNN.wav')
    parser.add_argument('--num_FFT',            type=int, default='512',    help='')
    parser.add_argument('--hop_size',           type=int, default='128',    help='')
    parser.add_argument('--window',             type=str, default='hamming',help='boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser')
    parser.add_argument('--r',                  type=int, default='40',    help='number of basis in NMF')
    parser.add_argument('--max_iter',           type=int, default='50',    help='number of maximum of NMF iteration')
    parser.add_argument('--display_step',       type=int, default='10',     help='display step in NMF interation')
    parser.add_argument('--p',                  type=int, default='2',      help='parameter in wiener filter for gain')
    parser.add_argument('--n_epoch', type=int, default='50', help='number of DNN epoch')
    parser.add_argument('--n_hidden', type=int, default='400', help='hidden units of DNN')
    parser.add_argument('--drop_out', type=float, default='1', help='dropout of DNN')
    parser.add_argument('--n_batch',    type=int, default='1024', help='mini batch size')

    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.datasets_dir):
        os.makedirs(args.datasets_dir)
    assert args.num_FFT >= 1, 'number of FFT size must be larger than or equal to one'
    assert args.hop_size < args.num_FFT, 'hop size must be smaller than number of FFT size'
    return args

if __name__ == '__main__':
    args = parse_args()
    NMF_DNN(args)
