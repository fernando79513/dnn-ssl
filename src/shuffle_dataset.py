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

    for _, row in data.iterrows():
        name = row[0]
        ftype = row[1]

        for id in range(100,5000,100):
            stft_data_file = f'{data_path}stft_data_{name}_{ftype}_{id}.npy'
            pmap_file = f'{data_path}pmap_{name}_{ftype}_{id}.npy'

            if not os.path.isfile(stft_data_file):
                print('no file')
                continue

            stft_data = np.load(stft_data_file)
            pmaps = np.load(pmap_file)

            print(stft_data.shape)
            print(pmaps.shape)

            stft_data_slim = stft_data[::10]
            pmaps_slim = pmaps[::10] 

            print(stft_data_slim.shape)
            print(pmaps_slim.shape)

            print('saving file!')
            stft_slim_file = f'{shuff_path}stft_data_{name}_{ftype}_{id}.npy'
            pmap_slim_file = f'{shuff_path}pmap_{name}_{ftype}_{id}.npy'
            np.save(stft_slim_file, stft_data_slim)
            np.save(pmap_slim_file, pmaps_slim)

            
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

def prune_dataset(params):
    
    data_path = f"data/{params['mic_array']['name']}/dataset/"
    shuff_path  = f"data/{params['mic_array']['name']}/dev/"

    data = pd.read_csv(f'{data_path}data.csv')

    for _, row in data.iterrows():
        name = row[0]
        ftype = row[1]


        pmap_list = []
        stft_data_list = []

        for id in range(100,5000,100):
            stft_data_file = f'{data_path}stft_data_{name}_{ftype}_{id}.npy'
            pmap_file = f'{data_path}pmap_{name}_{ftype}_{id}.npy'

            if not os.path.isfile(stft_data_file):
                print('no file')
                continue

            stft_data = np.load(stft_data_file)
            pmaps = np.load(pmap_file)

            print(stft_data.shape)
            print(pmaps.shape)

            stft_data_slim = stft_data[::10]
            pmaps_slim = pmaps[::10] 

            pmap_list.append()
            stft_data_list.append()

            print(stft_data_slim.shape)
            print(pmaps_slim.shape)

            print('saving file!')
            stft_slim_file = f'{shuff_path}stft_data_{name}_{ftype}_{id}.npy'
            pmap_slim_file = f'{shuff_path}pmap_{name}_{ftype}_{id}.npy'
            np.save(stft_slim_file, stft_data_slim)
            np.save(pmap_slim_file, pmaps_slim)

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


    shuffle_dataset(params)


    # simulate_speech(params)
    print('DONE')
