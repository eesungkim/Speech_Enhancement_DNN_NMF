# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:12:48 2018

@author: eesungkim
"""

import librosa
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
import os
import librosa
import argparse
from utils.utils import * 
from numpy.linalg import norm


def NMF(args):
    PATH_MATLAB='"C:/Program Files/MATLAB/R2014a/bin/matlab.exe"'
    PATH_ROOT = os.getcwd() 
    PATH_MATLAB1 = os.path.join(PATH_ROOT , 'PESQ_MATLAB/execute_pesq.m')
    
    os.chdir(PATH_ROOT)
    path_clean_train        = os.path.join(PATH_ROOT , args.input_clean_train)
    path_clean_test         = os.path.join(PATH_ROOT , args.input_clean_test)
    path_noisy_test         = os.path.join(PATH_ROOT , args.input_noisy_test)

    path_noise_1            = os.path.join(PATH_ROOT , args.input_noise_1)
    path_noise_2            = os.path.join(PATH_ROOT , args.input_noise_2)
    path_noise_3            = os.path.join(PATH_ROOT , args.input_noise_3)

    output_path_estimated_noisy_test = os.path.join(PATH_ROOT , args.output_file)
    
    (sr, clean_train)     = wav.read(path_clean_train)
    (sr, clean_test)      = wav.read(path_clean_test)
    (sr, noisy_test)      = wav.read(path_noisy_test)

    (sr, noise_1)         = wav.read(path_noise_1)
    (sr, noise_2)         = wav.read(path_noise_2)
    (sr, noise_3)         = wav.read(path_noise_3)

    clean_train             = clean_train.astype('int16')
    noisy_test              = noisy_test.astype('int16')
    noise_1                 = noise_1.astype('int16')
    noise_2                 = noise_2.astype('int16')
    noise_3                 = noise_3.astype('int16')


    # NMF training stage    
    #####################################################################################
    #clean_train
    stft_clean_train = librosa.stft(clean_train, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)   
    stft_clean_train = stft_clean_train[:,:10000]
    magnitude_clean_train, phase_clean_train = divide_magphase(stft_clean_train, power=1)
    #obtain the basis matrix of clean_speech
    W_clean_train, H_clean_train  = NMF_MuR(magnitude_clean_train,args.r,args.max_iter,args.display_step,const_W=False,init_W=0)

    # noise
    ##########################################################
    # 1) 각 노이즈 마다 3000 frame 씩 이어붙혀서 총 9000으로 만들어서 40 base 로 만들기

    # # three noises
    stft_noise_1 = librosa.stft(noise_1, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    stft_noise_2 = librosa.stft(noise_2, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    stft_noise_3 = librosa.stft(noise_3, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)

    magnitude_noise_1, _ = divide_magphase(stft_noise_1, power=1)
    magnitude_noise_2, _ = divide_magphase(stft_noise_2, power=1)
    magnitude_noise_3, _ = divide_magphase(stft_noise_3, power=1)

    magnitude_noise_1, magnitude_noise_2, magnitude_noise_3= magnitude_noise_1[:,:3000],magnitude_noise_2[:,:3000],magnitude_noise_3[:,:3000]
    nmf_magnitude_noise = np.concatenate((magnitude_noise_1, magnitude_noise_2, magnitude_noise_3),axis=1)
    #obtain the basis matrix of noise
    W_noise, H_noise              = NMF_MuR(nmf_magnitude_noise,args.r,args.max_iter,args.display_step,const_W=False,init_W=0)


    # # # 2) base 13,13,14로 이어 붙히기
    # stft_noise_1 = librosa.stft(noise_1, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    # stft_noise_2 = librosa.stft(noise_2, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    # stft_noise_3 = librosa.stft(noise_3, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)
    # stft_noise_1 = stft_noise_1[:,:9000]
    # stft_noise_2 = stft_noise_2[:,:9000]
    # stft_noise_3 = stft_noise_3[:,:9000]
    # magnitude_noise_1, _ = divide_magphase(stft_noise_1, power=1)
    # magnitude_noise_2, _ = divide_magphase(stft_noise_2, power=1)
    # magnitude_noise_3, _ = divide_magphase(stft_noise_3, power=1)
    #
    # #obtain the basis matrix of noise
    # W_noise_1, _ = NMF_MuR(magnitude_noise_1,13,args.max_iter,args.display_step,const_W=False,init_W=0)
    # W_noise_2, _ = NMF_MuR(magnitude_noise_2, 13, args.max_iter, args.display_step, const_W=False, init_W=0)
    # W_noise_3, _ = NMF_MuR(magnitude_noise_3, 14, args.max_iter, args.display_step, const_W=False, init_W=0)
    # ####################################################
    # W_noise = np.concatenate((W_noise_1, W_noise_2, W_noise_3),axis=1)

    #noisy
    stft_noisy_test = librosa.stft(noisy_test, n_fft=args.num_FFT, hop_length=args.hop_size, window=args.window)   
    magnitude_noisy_test, phase_noisy_test = divide_magphase(stft_noisy_test, power=1)
    #####################################################################################
    
    W_noisy = np.concatenate([W_clean_train,W_noise], axis=1)
    _,H_reconstructed_noisy = NMF_MuR(magnitude_noisy_test,2*args.r,args.max_iter,args.display_step,const_W=True, init_W=W_noisy)
    
    H_reconstructed_clean = H_reconstructed_noisy[:args.r,:]
    H_reconstructed_noise = H_reconstructed_noisy[args.r:,:]
    
    magnitude_reconstructed_clean=np.matmul(W_clean_train,H_reconstructed_clean)
    magnitude_reconstructed_noise = np.matmul(W_noise,H_reconstructed_noise)
     
    #Gain function similar to wiener filter to enhance the speech signal
    wiener_gain = np.power(magnitude_reconstructed_clean,args.p) / (np.power(magnitude_reconstructed_clean,args.p) + np.power(magnitude_reconstructed_noise, args.p))
    magnitude_estimated_clean = wiener_gain * magnitude_noisy_test

    #Reconstruct
    stft_reconstructed_clean = merge_magphase(magnitude_estimated_clean, phase_noisy_test)
    signal_reconstructed_clean =librosa.istft(stft_reconstructed_clean, hop_length=args.hop_size, window=args.window)
    signal_reconstructed_clean = signal_reconstructed_clean.astype('int16')
    wav.write(output_path_estimated_noisy_test,sr,signal_reconstructed_clean)
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
    print("[%s]\n Original: %.2f\n NMF\t: %.2f"%(name,noisy_original_PESQ,enhanced_PESQ))

    # print('Noisy STOI: %.6f' % calc_stoi(clean_test / norm(clean_test), noisy_test / norm(noisy_test), sr))
    # print('NMF STOI: %.6f' % calc_stoi(clean_test / norm(clean_test), signal_reconstructed_clean / norm(signal_reconstructed_clean), sr))

    # Display signals, spectrograms
    show_signal(clean_test,noisy_test,signal_reconstructed_clean,sr)
    show_spectrogram(clean_test,noisy_test, signal_reconstructed_clean, sr, args.num_FFT,args.hop_size)



    
def parse_args():
    parser = argparse.ArgumentParser(description='NMF Speech Enhancement')
    parser.add_argument('--datasets_dir',       type=str, default='datasets/')

    parser.add_argument('--input_clean_train',  type=str, default='datasets/timit_clean_selected_train_total.wav')
    parser.add_argument('--input_clean_test',   type=str, default='datasets/timit_clean_selected/timit_clean_selected_test.wav')
    parser.add_argument('--input_noisy_test',   type=str, default='datasets/timit_noisy_selected/test_match/timit_noisy_babble_snr10_test.wav')

    parser.add_argument('--input_noise_1', type=str, default='datasets/noise/NOISEX/babble.wav')
    parser.add_argument('--input_noise_2', type=str, default='datasets/noise/NOISEX/factory1.wav')
    parser.add_argument('--input_noise_3', type=str, default='datasets/noise/NOISEX/machinegun.wav')
    parser.add_argument('--output_file',        type=str, default='datasets/output/estimated_clean_NMF.wav')

    parser.add_argument('--num_FFT',            type=int, default='512',    help='')
    parser.add_argument('--hop_size',           type=int, default='128',    help='')
    parser.add_argument('--window',             type=str, default='hamming',help='boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser')
    parser.add_argument('--r',                  type=int, default='40',    help='number of basis in NMF')
    parser.add_argument('--max_iter',           type=int, default='50',    help='number of maximum of NMF iteration')
    parser.add_argument('--display_step',       type=int, default='10',     help='display step in NMF interation')
    parser.add_argument('--p',                  type=int, default='2',      help='parameter in wiener filter for gain')
    
    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.datasets_dir):
        os.makedirs(args.datasets_dir)
    assert args.num_FFT >= 1, 'number of FFT size must be larger than or equal to one'
    assert args.hop_size < args.num_FFT, 'hop size must be smaller than number of FFT size'
    return args

if __name__ == '__main__':
    args = parse_args()
    NMF(args)
