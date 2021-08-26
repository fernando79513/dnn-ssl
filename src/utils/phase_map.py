import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal

FS=16000
WINDOW="hann"
NPERSEG=400
NFFT=512
STRIDE=160
NOVERLAP=NPERSEG-STRIDE

def split_stft_mc(stft_mc, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, stft_mc.shape[2]//n):
        yield stft_mc[:, :, i*n:i*n + n]

def stft(wav):
    '''
    Performs multi_channel stft based on custom params
    '''
    
    stft_list = []
    for channel in wav.T:
        f, t, Zxx = signal.stft(channel, fs=FS, window=WINDOW, nperseg=NPERSEG,
            noverlap=NOVERLAP, nfft=NFFT)
        stft_list.append(Zxx)
    stft_mc = np.array(stft_list)
    return stft_mc

def plot_specgram(t, f, Zxx):
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=-20, vmax=30, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return

def plot_phasemap(t, f, Zxx):
    plt.pcolormesh(t, f, np.angle(Zxx), vmin=-3, vmax=3, shading='gouraud')
    plt.title('STFT Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return

if __name__ == "__main__":
    i = 20
    filename = f"simulations/matrix_voice/1_src/clean_{i:0>4}.wav"
    fs, wav = wavfile.read(filename)
    # Create the STFT object + set filter and appropriate zero-padding
    # stft = pra.realtime.STFT(400, hop=160, channels=1)

    # mono = wav[:,1]
    stft_mc = stft(wav)
    print(stft_mc.shape)

    input_labels  = ['id', 'chunk', 'stft']
    df = pd.DataFrame(columns=input_labels)
    row = {
        'id':'0001', 'chunk':2, 'stft':stft_mc[:,:,:20]
    }
    df = df.append(row, ignore_index=True)
    print(df)
    print(df['stft'][0].shape)
        
    # plot the spectrogram before and after filtering
    f, t, Zxx = signal.stft(wav.T[0], fs=fs, window="hann", nperseg=400, noverlap=240,
        nfft=512)
    print(Zxx.shape)
    print(np.mean(np.angle(Zxx)/(2*np.pi) + .5))
    print(np.max(np.angle(Zxx)/(2*np.pi) + .5))
    print(np.min(np.angle(Zxx)/(2*np.pi) + .5))
    plot_specgram(t, f, Zxx)
    plot_phasemap(t, f, Zxx)

    new_wav = signal.istft(Zxx, fs=fs, window="hann", nperseg=400, noverlap=240,
        nfft=512)
    new_wav = new_wav[1].astype(np.int16)

    import sounddevice as sd
    print('playing sound using  pydub')
    sd.play(new_wav, fs)
    # sd.play(mono, fs)
    sd.wait()