from operator import ne
from scipy.io import wavfile
from scipy import signal
import numpy as np
import pandas as pd
import ast
import json
import os
from xml.dom import minidom
import matplotlib.pyplot as plt
from scipy.signal.spectral import stft

from scipy.signal.windows.windows import cosine
from tensorflow.core.framework.types_pb2 import DT_UINT16
from tensorflow.python.keras.backend import dtype
# import h5py

from src.utils import logs
from src.utils import phase_map

# 0201,2,0,
# bdl,[295.           2.53046924   1.3828238 ],"['b0399', 'b0471', 'a0333']",
# awb,[170.           1.42061424   1.04998708],"['a0459', 'b0204', 'b0394']",
# [6.7858    7.0624576 3.289738 ],0.9075,
# matrix_voice,[4.95621104 2.91635804 1.8599885 ]


R = 0.03829
C = 343
f_array = np.linspace(0,8000,257)
fs =16000
T = 500
F = 257

T_ARRAY = np.linspace(0, 0.019, F)
N = 2
R = 0.03829
C = 343
FS = 16000
F_ARRAY = np.linspace(0,8000,F)
MIC_ANGLES = [
     3.0479777621063042,  2.1505430344911307,  1.2529265224648303,
     0.3551953279695166, -0.5422998705472181, -1.4398667411390826,
    -2.337544759805849  ]
M = len(MIC_ANGLES)

def get_steering_vectors(theta, mic_angles):
    # todo change for tensors
    tau = np.empty((N, M))
    d   = np.empty((N, M, F), dtype=np.complex)
    for n in range(N):
        for m in range(M):
            tau[n,m] = R/C * np.cos(theta[n] - mic_angles[m])
            for f, freq in enumerate(F_ARRAY):
                d[n,m,f] = np.exp(-1j*2*np.pi*freq*tau[n,m])
    return d

def get_angle_features(d, y):
    # todo change for tensors
    a = np.empty((N, F, T))
    a_pred = np.empty((N, F, T))
    for n in range(N):
        for f in range(F):
            for t in range(T):
                a_pred[n,f,t] = np.abs(np.dot(d[n,:,f].conj(), y[:,f,t]))**2

    #  TODO: check first paper
    for n in range(N):
        for f in range(F):
            for t in range(T):
                for s in range(N):
                    a[n,f,t] = a_pred[n,f,t]
                    if s == n:
                        pass
                    else:
                        if a_pred[n,f,t] < a_pred[s,f,t]:
                            a[n,f,t] = 0
    return a

def get_scm(l, y):
    scm = np.empty((N, F, M, M), dtype=complex)
    for n in range(N):
        for f in range(F):
            mask_mat = np.zeros((M, M), dtype=complex)
            for t in range(T):
                y_arr = np.array([y[:,f,t]]).T
                mask_mat += l[n,f,t]*np.matmul(y_arr, y_arr.conj().T)
            scm[n,f,:,:] = 1/np.sum(l[n,f,:])*mask_mat
    return scm

def get_ref_vector(i):
    u = np.zeros(M)
    u[i] = 1
    return u

def get_mvdr_ref(scm, u):
    b = np.empty((N, F, M), dtype=complex)
    scm_i = np.empty((F, M, M), dtype=complex)
    for n in range(N):
        for f in range(F):
            for s in range(N):
                if s == n:
                    scm_n = scm[s]
                else:
                    scm_i += scm[s]
            scm_mult = np.matmul(np.linalg.inv(scm_i[f]), scm_n)
            b[n,f,:] = np.matmul(scm_mult[f,:,:]/np.trace(scm_mult[f,:,:]), u)
    return b

def stft_t(wav):
    '''
    Performs multi_channel stft based on custom params
    '''
    
    stft_list = []
    for channel in wav.T:
        f, t, Zxx = signal.stft(channel, fs=fs,
            nperseg=400, noverlap=240, nfft=512)
        stft_list.append(Zxx)
    stft_mc = np.array(stft_list)
    return stft_mc, t

class MicArray:
    def __init__(self, name):
        self.name = name
        self.mic_count = 0
        self.positions = np.zeros((2,8))
        self.x_pos = np.zeros(8)
        self.y_pos = np.zeros(8)

        self.get_mics(name)

    def get_mics(self, name='matrix_voice'):
        # Read the microphone array
        # parsee an xml file by name
        mydoc = minidom.parse(f'config/{name}.xml')
        items = mydoc.getElementsByTagName('position')
        self.mic_count = len(items)
        self.x_pos = np.zeros(self.mic_count)
        self.y_pos = np.zeros(self.mic_count)

        for i, elem in enumerate(items):  
            self.x_pos[i] = float(elem.attributes['x'].value)
            self.y_pos[i] = float(elem.attributes['y'].value)

        self.positions = np.array([self.x_pos, self.y_pos])
        return

def get_features():
    data_path = f"data/matrix_voice/test/"
    stfts = np.load(f'{data_path}stft_2_src_clean.npy')
    stft_data = np.load(f'{data_path}stft_data_2_src_clean.npy')

    index = 8

    mask = stft_data[:, 0] == 0
    test_data = stft_data[mask,:][index-1]
    test_stft = stfts[mask,:][index-1]


    filename = f"simulations/matrix_voice/test/2_src/clean_{index:0>4}.wav"
    fs, wav = wavfile.read(filename)

    stft_mc, times = stft_t(wav)
    print(stfts[0].shape)
    print(stft_data[0].shape)

    test_angles_deg = test_data[3:5]
    test_angles = np.deg2rad(test_data[3:5])
    print(test_stft.shape)
    print(test_angles)
    print(test_angles_deg)

    mic_array = MicArray('matrix_voice')
    mic_angles = []
    mic_angles_deg = []
    for i, position in enumerate(mic_array.positions.T):
        if i == 0:
            continue
        mic_angle = np.arctan2(position[1], position[0])

        mic_angles.append(mic_angle)
        mic_angles_deg.append(np.rad2deg(mic_angle))

    print(mic_angles[0])
    print(mic_angles)
    print(test_angles[0])
    print(test_angles[0]-mic_angles[0])

    taus_0 = []
    taus_1 = []
    for angle in mic_angles: 
        tau_0 = R/C*np.cos(angle-test_angles[0])
        tau_1 = R/C*np.cos(angle-test_angles[1])
        taus_0.append(tau_0)
        taus_1.append(tau_1)
    print(taus_0)
    print(taus_1)

    d_1 = np.empty((F, 7), dtype = np.complex)
    d_2 = np.empty((F, 7), dtype = np.complex)
    for f, freq in enumerate(f_array):
        d_1[f,:] = np.array([np.exp(-1j*2*np.pi*freq*tau) for tau in taus_0])
        d_2[f,:] = np.array([np.exp(-1j*2*np.pi*freq*tau) for tau in taus_1])



    stft_mc = stft_mc[1:, :, :]
    print(stft_mc.shape)
    # stft_mc = np.moveaxis(stft_mc, 0, 1)
    a_1 = np.empty((F,T))
    a_2 = np.empty((F,T))     
    
    norm_d_1 = np.conjugate(d_1).T
    norm_d_2 = np.conjugate(d_2).T

    for t in range(T):
        for f in range(F):
            a_1[f,t] = np.abs(np.dot(d_1[f,:], (stft_mc[:,f,t])))
            a_2[f,t] = np.abs(np.dot(d_2[f,:], (stft_mc[:,f,t])))

    print(type(stft_mc))

    print(a_1.shape)
    print(times.shape)
    print(f_array.shape)
    plt.pcolormesh(times[:500], f_array, a_2, vmin=-20, vmax=30, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    print(stft_mc[0].shape)
    # Zxx = a_1[3]

    new_wav = signal.istft(a_2, fs=16000, nperseg=400, noverlap=240,
        nfft=512)
    new_wav = new_wav[1].astype(np.int16)
    print(np.max(new_wav))

    print(wav.shape)
    mono = wav[:,1]
    # phase_map.plot_specgram(t , f , np.abs(a_1[1]))
    import sounddevice as sd
    print('playing sound using  pydub')
    sd.play(new_wav, fs)
    # sd.play(mono, fs)
    sd.wait()

    return

if __name__ == '__main__':
    get_features()
