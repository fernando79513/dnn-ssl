import numpy as np


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc

def prepare_input(wav, mic_pairs, length, max_tau, interp):

    fpt_list = np.empty((0,3))
    cc_len = (max_tau*interp*2)+1
    cc_list  = np.empty((0,cc_len))
    
    frames = len(wav)//length
    
    for i in range(frames):
        for j, pair in enumerate(mic_pairs): 
            a_id = pair[0]
            b_id = pair[1]
            tau, cc =  gcc_phat(wav.T[a_id, i*length:(i+1)*length-1],
                wav.T[b_id, i*length:(i+1)*length-1], interp=4, max_tau=9)
            fpt_list = np.append(fpt_list, [[i, j, tau]], axis=0)
            cc_list = np.append(cc_list,[cc],axis=0)
    
    return fpt_list, cc_list


if __name__ == "__main__":

    refsig = np.linspace(1, 10, 10)

    for i in range(0, 10):
        sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
        offset, _ = gcc_phat(sig, refsig)
        print(offset)





