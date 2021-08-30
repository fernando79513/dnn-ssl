from scipy.io import wavfile
import numpy as np
import pandas as pd
import ast
import json
import os

from scipy.signal.spectral import stft
# import h5py


from src.utils import logs
# from src.utils import csv_utils
from src.utils import phase_map

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


def prune_dataset(params):

    # if params['test']:
    #     data_path = f"data/{params['mic_array']['name']}/test/"
    #     shuff_path  = f"data/{params['mic_array']['name']}/test/"
    # else:
    #     data_path = f"data/{params['mic_array']['name']}/dataset/"
    #     shuff_path  = f"data/{params['mic_array']['name']}/dev/"

    data_path = f"data/{params['mic_array']['name']}/dataset/"
    shuff_path  = f"data/{params['mic_array']['name']}/dev/"

    data = pd.read_csv(f'{data_path}data.csv')

    phasemaps = []
    stft_data = []
    for _, row in data.iterrows():
        name = row[0]
        ftype = row[1]

        for id in range(0,4000,100):
            stft_file = f'{data_path}stft_data_{name}_{ftype}_{id}.npy'
            pmap_file = f'{data_path}pmap_{name}_{ftype}_{id}.npy'

            if not os.path.isfile(stft_file):
                print('no file')
                continue

            np.load(stft_file, pmap_file)

            print(stft_file.shape)
            print(pmap_file.shape)
            return
            
            
        #     if id % 100 == 0:
        #         print('saving file!')
        #         stft_file = f'{data_path}stft_data_{name}_{ftype}_{id}.npy'
        #         pmap_file = f'{data_path}pmap_{name}_{ftype}_{id}.npy'
        #         stft_data_np = np.array(stft_data)
        #         p_map_np = np.array(phasemaps)
        #         np.save(stft_file, stft_data_np)
        #         np.save(pmap_file, p_map_np)
        #         phasemaps = []
        #         stft_data = []

        # if params['test']:
        #     print('saving file!')
        #     stft_file = f'{data_path}stft_data_{name}_{ftype}.npy'
        #     pmap_file = f'{data_path}pmap_{name}_{ftype}.npy'
        #     stft_data_np = np.array(stft_data)
        #     p_map_np = np.array(phasemaps)
        #     np.save(stft_file, stft_data_np)
        #     np.save(pmap_file, p_map_np)
        #     phasemaps = []
        #     stft_data = []        
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


    prune_dataset(params)


    # simulate_speech(params)
    print('DONE')
