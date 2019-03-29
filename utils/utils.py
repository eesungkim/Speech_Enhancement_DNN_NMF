# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:43:28 2018
@author: eesungkim
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io.wavfile as wav

def cost_snmf(V, W, H, beta=2,mu=0.1):
    A=tf.matmul(W,H)
    tmp=W*tf.matmul(A**(beta-1), H.T)
    numerator  = tf.matmul(A**(beta-2)*V,H.T) + W*(tf.matmul(tf.ones((tf.shape(tmp)[0],tf.shape(tmp)[0])),tmp )) 
    tmp2=W*tf.matmul(A**(beta-2)*V,H.T)
    denumerator= tf.matmul(A**(beta-1),H.T) + W*(tf.matmul(tf.ones((tf.shape(tmp2)[0],tf.shape(tmp2)[0])),tmp2))
    W_new = numerator/denumerator
    
    H_new= tf.matmul(W.T,V*A**(beta-2))/(tf.matmul(W.T,A**(beta-1))+mu)
    
    return W_new, H_new

def optimize(mode, V, W, H, beta, mu, lr, const_W):
    #cost function
    cost = tf.reduce_mean(tf.square(V - tf.matmul(W, H)))

    #update operation for H
    if mode=='snmf':
        #Sparse NMF MuR
        
        A=tf.matmul(W,H)
        H_new= tf.matmul(tf.transpose(W),V*A**(beta-2))/(tf.matmul(tf.transpose(W),A**(beta-1))+mu)
        H_update = H.assign(H_new)
    elif mode=='nmf':
        #Basic NMF MuR
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
        H_update = H.assign(H_new)
    elif mode=='pg':
        """optimization; Projected Gradient method """
        dW, dH = tf.gradients(xs=[W, H], ys=cost)
        H_update_ = H.assign(H - lr * dH)
        H_update = tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_)

    #update operation for W 
    if const_W == False:
        if mode=='snmf':
            #Sparse NMF MuR

            vec = tf.reduce_sum(W,0)
            multiply = tf.constant([tf.shape(W)[0]])

            de=tf.reshape(tf.tile(tf.reduce_sum(W,0), multiply), [ multiply[0], tf.shape(tf.reduce_sum(W,0))[0]])
            W=W/de
            Ht=tf.transpose(H)
            tmp=W*tf.matmul(A**(beta-1), Ht)
            n=tf.shape(tmp)[0]
            numerator  = tf.matmul(A**(beta-2)*V,Ht) + W*(tf.matmul(tf.ones((n,n)),tmp )) 
            tmp2=W*tf.matmul(A**(beta-2)*V,Ht)
            denumerator= tf.matmul(A**(beta-1),Ht) + W*(tf.matmul(tf.ones((n,n)),tmp2))
            W_new = W*numerator/denumerator
            W_update = W.assign(W_new)
            ###############################################################################
            # Ht=tf.transpose(H)
            # tmp=W*tf.matmul(A**(beta-1), Ht)
            # n=tf.shape(tmp)[0]
            # numerator  = tf.matmul(A**(beta-2)*V,Ht) 
            # tmp2=W*tf.matmul(A**(beta-2)*V,Ht)
            # denumerator= tf.matmul(A**(beta-1),Ht)
            # W_new = W*numerator/denumerator
            # W_update = W.assign(W_new)
        elif mode=='nmf':
            #Basic NMF MuR
            Ht = tf.transpose(H)
            W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
            W_update = W.assign(W_new)
        elif mode=='pg':
            W_update_ = W.assign(W - lr * dW)
            W_update = tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_)



        return W_update,H_update, cost

    return 0, H_update, cost







# code from https://github.com/eesungkim/NMF-Tensorflow
def NMF_MuR(V_input,r,max_iter,display_step,const_W,init_W):
    m,n=np.shape(V_input)
    
    tf.reset_default_graph()
    
    V = tf.placeholder(tf.float32) 
    
    initializer = tf.random_uniform_initializer(0,1)

    if const_W==False:
        W =  tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H =  tf.get_variable("H", [r, n], initializer=initializer)
    else:
        W =  tf.constant(init_W, shape=[m, r], name="W")
        H =  tf.get_variable("H", [r, n], initializer=initializer)
        
    mode='nmf'
    W_update, H_update, cost=optimize(mode, V, W, H, beta=2, mu=0.00001, lr=0.1, const_W=const_W)


    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        for idx in range(max_iter):
            if const_W == False:
                W=sess.run(W_update, feed_dict={V:V_input})
                H=sess.run(H_update, feed_dict={V:V_input})
            else:
                H=sess.run(H_update, feed_dict={V:V_input})
                
            if (idx % display_step) == 0:
                costValue = sess.run(cost,feed_dict={V:V_input})
                print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue/1))

    print("================= [Completed Training NMF] ===================")
    return W, H

def NMF_MuR_H(V_input,r,max_iter,display_step,const_H,init_H):
    m,n=np.shape(V_input)
    
    tf.reset_default_graph()
    
    V = tf.placeholder(tf.float32) 
    
    initializer = tf.random_uniform_initializer(0,1)

    if const_H==False:
        W =  tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H =  tf.get_variable("H", [r, n], initializer=initializer)
    else:
        W =  tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H =  tf.constant(init_H, shape=[r, n], name="H")
        
    WH =tf.matmul(W, H)

    #cost function
    cost = tf.reduce_mean(tf.square(V - WH))

    #optimization; Multiplicative update Rule (MuR)
    #update operation for H
    if const_H == False:
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
        H_update = H.assign(H_new)
    
    #update operation for H 
    Ht = tf.transpose(H)
    W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
    W_update = W.assign(W_new)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        for idx in range(max_iter):
            costValue = sess.run(cost,feed_dict={V:V_input})
            if const_H == False:
                W=sess.run(W_update, feed_dict={V:V_input})
                H=sess.run(H_update, feed_dict={V:V_input})
            else:
                W=sess.run(W_update, feed_dict={V:V_input})
                
            if (idx % display_step) == 0:
                
                print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue/1000000))
    print("================= [Completed Training NMF] ===================")
    return W, H



def spectrogram(signal, NFFT, hop_length, window='hann'):
    return np.abs(librosa.stft(signal, n_fft=NFFT, hop_length=hop_length, window=window))**2

def LogPowerSpectrum(signal, NFFT, hop_length, window='hann'):
    return np.log(spectrogram(signal, NFFT, hop_length, window=window)) 

def show_signal(signal,signal2,signal_reconstructed,sr):
    plt.figure(figsize=(10,10))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(signal, sr=sr)
    plt.title('Clean Time Signal')

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(signal2, sr=sr)
    plt.title('Noisy Time Signal')

    plt.subplot(3, 1, 3)
    librosa.display.waveplot(signal_reconstructed, sr=sr)
    plt.title('Reconstructed Clean Time Signal')

def show_spectrogram(signal,signal2,recnstrtSignal,sr,NFFT,hop_length):
    #Display power (energy-squared) spectrogram
    plt.figure(figsize=(10,10))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(origianlSpectrogram, ref_power=np.max),sr=sr, x_axis='time',y_axis='linear')
    plt.title('Clean Spectrogram')
    plt.colorbar(format='%+02.0f dB')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=signal2, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(origianlSpectrogram, ref_power=np.max),sr=sr, x_axis='time',y_axis='linear')
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+02.0f dB')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=recnstrtSignal, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(recnstrtSpectrogram, ref_power=np.max),sr=sr, x_axis='time', y_axis='linear')
    plt.title('Reconstructed Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
def divide_magphase(D, power=1):
    """Separate a complex-valued stft D into its magnitude (S)
    and phase (P) components, so that `D = S * P`."""

    mag = np.abs(D)**power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def merge_magphase(magnitude, phase):
    """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
    # magnitude * np.exp(np.angle(D)*1j)
    # magnitude =  np.exp(magnitude)
    # magnitude = np.sqrt(magnitude)
    # magnitude * numpy.cos(np.angle(D))+ magnitude * numpy.sin(np.angle(D))*1j
    return magnitude * phase


def normalize_Zscore(X_train, X_test):
    #각 fre bin에서, 모든 X_train, X_test 에 대해서 norm
    X=np.concatenate((X_train, X_test), axis=1)
    mu = np.mean(X, axis = 1)
    sigma = np.std(X, axis = 1)

    mu1=np.array([mu,]*X_train.shape[1]).T
    sigma1=np.array([sigma,]*X_train.shape[1]).T

    X_train = (X_train - mu1) / sigma1
    X_test = (X_test - mu1) / sigma1
    return X_train, X_test

def normalize_MinMax(X_train, X_test):
    #각 fre bin에서, 모든 X_train, X_test 에 대해서 norm
    X=np.concatenate((X_train, X_test), axis=1)
    X_min = np.min(X, axis = 1)
    X_max = np.max(X, axis = 1)

    X_min1=np.array([X_min,]*X_train.shape[1]).T
    X_max1=np.array([X_max,]*X_train.shape[1]).T

    X_train = (X_train - X_min1) / (X_max1-X_min1)
    X_test = (X_test - X_min1) / (X_max1-X_min1) 
    return X_train, X_test

def nfft(frame_length):
    fft_size = 2
    while fft_size < frame_length:
        fft_size = fft_size * 2
    return fft_size

def apply_context_window(feats_mat, left_context, right_context):
    num_frames, num_bins = feats_mat.shape
    context = left_context + right_context + 1
    dumps_specs = np.zeros([num_frames, context * num_bins])
    for t in range(num_frames):
        for c in range(context):
            base = c * num_bins
            idx = t + c - left_context
            if idx < 0:
                idx = 0
            if idx > num_frames - 1:
                idx = num_frames - 1
            dumps_specs[t, base: base + num_bins] = feats_mat[idx]
    return dumps_specs.astype(np.float32)


def perform_stft(path,args):
    (sr, time_signal) = wav.read(path)
    time_signal=time_signal.astype('float')
    fft_window = nfft(args.frame_length)
    freq_signal = librosa.stft(time_signal, n_fft=fft_window, win_length=args.frame_length, hop_length=args.hop_size, window=args.window)
    magnitude, phase = divide_magphase(freq_signal, power=1)
    return magnitude, phase, sr
