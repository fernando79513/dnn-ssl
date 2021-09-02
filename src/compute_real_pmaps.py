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

    real_df = pd.read_csv(f"data/real/data_real.csv")
    data_path =f"data/real/"

    phasemaps = []
    stft_data = []

    for _, row in real_df.iterrows():
        print(row)
        n_src = row['n src']
        if n_src == '1_src':
            n = 1
        elif n_src == '2_src':
            n = 2
        ftype = row['type']
        if ftype == 'clean':
            n_noise = 0
        elif ftype == 'noise':
            n_noise = 1
        name = row['name']
        index = row['index']
        real_wav  = f"real_audio/matrix_voice/{ftype}_{name}_{index:0>4}.wav"

        wav = wavfile.read(real_wav)[1]
        print("processing ", real_wav)

        stft_mc = phase_map.stft(wav)
        chunks = []

        for i, chunk in enumerate(phase_map.split_stft_mc(stft_mc, 20)):
            # stft_data[id] = [
            #   0- index, 
            #   1- chunk_i, 
            #   2- number of speakers, 
            #   3- number of noises, 
            #   4- speaker angle 0, 
            #   5- speaker angle 1
            #   ]
            stft_data.append(np.array([index, i, n, n_noise, row['speaker angle 0'],
                row['speaker angle 1']]))
            p_map = np.angle(chunk)/(2*np.pi) + .5
            phasemaps.append(p_map)
            chunks.append(chunk)

    print('saving file!')
    pmap_file = f'{data_path}pmap_real.npy'
    p_map_np = np.array(phasemaps)
    np.save(pmap_file, p_map_np)
    stft_file = f'{data_path}stft_data_real.npy'
    stft_data_np = np.array(stft_data)
    np.save(stft_file, stft_data_np)
    phasemaps = []
    stft_data = []        
    return

if __name__ == '__main__':

    logs.set_logs()

    import argparse
    parser = argparse.ArgumentParser(
        description='Simulation of speeches inside a room')
    parser.add_argument('-c', '--config_file', type=str)

    args = parser.parse_args()

    if args.config_file == None:
         args.config_file = 'config/param_matrix_voice.json'
    with open(args.config_file) as f:
        params = json.load(f)


    compute_stft(params)

    # simulate_speech(params)
    print('DONE')
