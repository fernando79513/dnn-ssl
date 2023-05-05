import numpy as np
import tensorflow as tf


# import wandb
# from wandb.keras import WandbCallback

F = 257
# T = 20
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
    I = theta.shape[0]
    tau = np.empty((I, N, M))
    d   =  np.empty((I, N, M, F), dtype=np.complex)
    for n in range(N):
        for m in range(M):
            tau[:,n,m] = R/C * tf.cos(theta[:,n] - mic_angles[m])
    for f, freq in enumerate(F_ARRAY):
        d[:,:,:,f] = np.exp(-1j*2*np.pi*freq*tau[:,:,:])
    return d

def get_angle_features(d, y):
    # todo change for tensors
    I = d.shape[0]
    T = y.shape[-1]
    a = np.empty((I, N, F, T))
    a_pred = np.empty((I, N, F, T))
    for i in range(I):
        for n in range(N):
            for f in range(F):
                for t in range(T):
                    a_pred[i,n,f,t] = np.abs(
                        np.dot(d[i,n,:,f].conj().T, y[i,:,f,t]))**2

    #  TODO: check first paper
    for i in range(I):
        for n in range(N):
            for f in range(F):
                for t in range(T):
                    for s in range(N):
                        a[i,n,f,t] = a_pred[i,n,f,t]
                        if s == n:
                            pass
                        else:
                            if a_pred[i,n,f,t] < a_pred[i,s,f,t]:
                                a[i,n,f,t] = 0
    return a

def separation(a):
    k = 0.5
    I = a.shape[0]
    N = a.shape[1]
    F = a.shape[2]
    T = a.shape[3]
    l = np.zeros((I, N, F, T))
    v = np.zeros((I, N, F, T))

    for i in range(I):
        for f in range(F):
            for t in range(T):
                v[i,:,f,t] = tf.nn.softmax(a[i,:,f,t])
    l = 1./(1.-k)*tf.nn.relu(v-k)
    return l

def get_scm(l, y):
    I = l.shape[0]
    T = y.shape[-1]
    scm = np.empty((I, N, F, M, M), dtype=complex)
    for i in range(I):
        for n in range(N):
            for f in range(F):
                mask_mat = np.zeros((M, M), dtype=complex)
                if np.sum(l[i,n,f,:]) == 0:
                    scm[i,n,f,:,:] == np.zeros((M, M),dtype=complex)
                else:
                    for t in range(T):
                        y_arr = np.array([y[i,:,f,t]], dtype=complex).T
                        mask_mat += l[i,n,f,t]*np.matmul(y_arr, y_arr.conj().T)
                    scm[i,n,f,:,:] = 1/np.sum(l[i,n,f,:])*mask_mat
    return scm

def get_ref_vector(i):
    # TODO change this if we want ref variable
    u = np.zeros(M)
    u[i] = 1
    return u

def get_mvdr_ref(scm, u):
    I = scm.shape[0]
    b = np.empty((I, N, F, M), dtype=complex)
    scm_i = np.empty((I, F, M, M),dtype=complex)

    for i in range(I):
        for f in range(F):
            for n in range(N):
                for s in range(N):
                    if s == n:
                        scm_n = scm[i,s]
                    else:
                        scm_i += scm[i,s]
            if np.any(scm_i[i,f]):
                scm_mult = tf.matmul(tf.linalg.inv(scm_i[i,f]), scm_n[f])
                b[i,n,f,:] =np.matmul(scm_mult/np.trace(scm_mult), u)
            else:
                b[i,n,f,:] = np.zeros(M)
    return b

def sep_signal(b, y):
    I = b.shape[0]
    T = y.shape[-1]
    x = np.empty((I, N, F, T), dtype=complex)
    for i in range(I):
        for n in range(N):
            for f in range(F):
                for t in range(T):
                    x[i,n,f,t] = np.matmul(b[i,n,f,:].conj(), np.array([y[i,:,f,t]]).T)
    return x



from scipy.io import wavfile
from scipy import signal

def stft_t(wav):
    '''
    Performs multi_channel stft based on custom params
    '''
    stft_list = []
    for channel in wav.T:
        f, t, Zxx = signal.stft(channel, fs=16000,
            nperseg=400, noverlap=240, nfft=512)
        stft_list.append(Zxx)
    stft_mc = np.array(stft_list)
    return stft_mc, t


if __name__ == "__main__":

    x = np.load("tests/sep_sftt.npy")
    sep_spec_1 = x[:,0,...]
    sep_spec_2 = x[:,1,...]
    sep_spec_1 = np.concatenate(sep_spec_1, axis = -1)
    sep_spec_2 = np.concatenate(sep_spec_2, axis = -1)
    print(sep_spec_1.shape)
    import pdb;pdb.set_trace()

    sep_1 = signal.istft(sep_spec_1, fs=16000, window="hann", nperseg=400, noverlap=240,
        nfft=512)
    sep_1 = sep_1[1].astype(np.int16)
    sep_2 = signal.istft(sep_spec_2, fs=16000, window="hann", nperseg=400, noverlap=240,
        nfft=512)
    sep_2 = sep_2[1].astype(np.int16)

    # import pdb;pdb.set_trace()
    import sounddevice as sd
    print('playing sound using  pydub')
    sd.play(sep_2, samplerate=16000)
    # sd.play(mono, fs)
    sd.wait()


