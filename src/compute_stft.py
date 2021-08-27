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


def compute_stft(params):

    if params['test']:
        sim_path  = f"simulations/{params['mic_array']['name']}/test/"
        data_path = f"data/{params['mic_array']['name']}/test/"
    else:
        sim_path  = f"simulations/{params['mic_array']['name']}/"
        data_path = f"data/{params['mic_array']['name']}/"

    data = pd.read_csv(f'{data_path}data.csv')

    if os.path.isfile(f'{data_path}stft_data.csv'):
        df = pd.read_csv(f'{data_path}stft_data.pkl')
        last_row = df.iloc[-1]
    else:
        labels  = ['id', 'chunk', 'number of speakers', 'number of noises',
            'speaker angle 0', 'speaker angle 1', 'stft']
        df = pd.DataFrame(columns=labels)
        last_row = None

    phasemaps = []
    stft_data = []
    for _, row in data.iterrows():
        name = row[0]
        ftype = row[1]

        last_i = -1
        for i in range(50):
            if os.path.isfile(f'{data_path}stft_data_{name}_{ftype}_{i}.npy'):
                last_i = i*100

        simulation_df = pd.read_csv(f'{sim_path}{name}/positions_{ftype}.csv')
        for id in simulation_df['id']:
            if i < last_i:
                continue
            sim = simulation_df.iloc[id-1]
            n_speakers = sim['number of speakers']
            n_noises = sim['number of noises']

            wav_file = f'{ftype}_{id:0>4}.wav'
            wav = wavfile.read(f'{sim_path}{name}/{wav_file}')[1]
            logs.log_simulation(wav_file, n_speakers, n_noises, params)

            stft_mc = phase_map.stft(wav)
            angles, lengths = get_sim_info(sim)
            for i, chunk in enumerate(phase_map.split_stft_mc(stft_mc, 20)):
                sample_i = 400 + ((i+1) * 160 * 20)
                for j, length in enumerate(lengths):
                    if length < sample_i:
                        angles[j] = -1
                # row = { 'id':f'{id:0>4}', 
                #         'chunk':i,
                #         'number of speakers':n_speakers,
                #         'number of noises': n_noises,
                #         'speaker angle 0': angles[0],
                #         'speaker angle 1': angles[1]}
                # df = df.append(row, ignore_index=True)
                stft_data.append(np.array([i, n_speakers, n_noises, angles[0],
                    angles[1]]))
                p_map = np.angle(chunk)/(2*np.pi) + .5
                phasemaps.append(p_map)
            
            print('id is', id)
            if id % 100 == 0:
                print('saving file!')
                stft_file = f'{data_path}stft_data_{name}_{ftype}_{id}.npy'
                pmap_file = f'{data_path}pmap_{name}_{ftype}_{id}.npy'
                stft_data_np = np.array(stft_data)
                p_map_np = np.array(phasemaps)
                np.save(stft_file, stft_data_np)
                np.save(pmap_file, p_map_np)
                phasemaps = []
                stft_data = []




    # df.to_pickle(f'{data_path}stft.pkl')
    # if params['test']:
    #     df.drop(columns=['stft']).to_csv(f'{data_path}stft_data.csv')
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
