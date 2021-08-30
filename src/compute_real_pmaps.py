from numpy.core.function_base import linspace
from scipy.io import wavfile
import numpy as np
import pandas as pd
import ast
import json
import os


from src.utils import logs
# from src.utils import csv_utils
from src.utils import phase_map
from scipy import signal
import matplotlib.pyplot as plt

def get_sim_info(df):

    wav_path = 'wav/cmu_arctic/'
    n_speakers =  df['number of speakers']
    if 0 <= n_speakers <= 2:
        angles = [-1, -1]
        lengths = [-1, -1]
        for i in range(n_speakers):
            angle = np.fromstring(df[f'speaker position {i}'][1:-1],
                dtype=int, sep=' ')[0]
            angles[i] = angle

            audio_length =0
            name = df[f'speaker name {i}']
            audios = ast.literal_eval(df[f'speaker audios {i}'])
            for audio in audios:
                wav_file = f'{wav_path}{name}/arctic_{audio}.wav'
                wav = wavfile.read(wav_file)[1]
                audio_length += len(wav)
            lengths[i] = audio_length
    else:
        print("Wrong number of speakers")
    return angles, lengths


def compute_stft(params):

    real_wav  = f"real_audio/2_sources/recording_voice_2src_clean_1.wav"
    data_path = f"data/real/"

    phasemaps = []
    stft_data = []

    wav = wavfile.read(real_wav)[1]
    print("processing ", real_wav)

    stft_mc = phase_map.stft(wav[:][:6000000])
    chunks = []
    t = linspace(1,10,2000)
    f_array = linspace(1,800,257)
    for i, chunk in enumerate(phase_map.split_stft_mc(stft_mc, 20)):
        # stft_data[id] = [
        #   0- chunk_i, 
        #   1- number of speakers, 
        #   2- number of noises, 
        #   3- speaker angle 0, 
        #   4- speaker angle 1
        #   ]
        stft_data.append(np.array([i, 2, 0, 30,
            70]))
        p_map = np.angle(chunk)/(2*np.pi) + .5
        phasemaps.append(p_map)
        chunks.append(chunk)
        print(i)
        if (i+1) % 100 == 0:
            specgram = np.concatenate(chunks, axis=2)
            # import pdb;pdb.set_trace()
            plt.pcolormesh(t, f_array, np.abs(specgram[0,:,:]), vmin=-20, vmax=30, shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
            chunks = []


    # print('saving file!')
    # stft_file = f'{data_path}stft_data_real.npy'
    # pmap_file = f'{data_path}pmap_real.npy'
    # stft_data_np = np.array(stft_data)
    # p_map_np = np.array(phasemaps)
    # np.save(stft_file, stft_data_np)
    # np.save(pmap_file, p_map_np)
    # phasemaps = []
    # stft_data = []        
    return

if __name__ == '__main__':

    logs.set_logs()

    import argparse
    parser = argparse.ArgumentParser(
        description='Simulation of speeches inside a room')
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-s', '--num_speakers', type=int)
    parser.add_argument('-n', '--num_noises', type=int)
    parser.add_argument('-t', '--test', type=bool, nargs='?', const=True)
    args = parser.parse_args()

    if args.config_file == None:
         args.config_file = 'config/param_matrix_voice.json'
    with open(args.config_file) as f:
        params = json.load(f)

    if args.num_speakers != None:
        params['speakers']['count'] = args.num_speakers
    if args.num_noises != None:
        params['noises']['count'] = args.num_noises
    if args.test != None:
        params['test'] = args.test


    compute_stft(params)


    # simulate_speech(params)
    print('DONE')
