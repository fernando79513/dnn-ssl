

# # Data Preparation

import logging
from operator import length_hint
from numpy.lib.utils import info
from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os


from src.utils import logs
from src.utils import csv_utils
from src.utils import gcc_phat
from src.utils.labels import LabelGenerator


# def input_to_h5(hf, index, fpt_list, cc_list):
#     ifpt_list = np.hstack((np.full((len(fpt_list), 1), index), fpt_list))
#     hf.create_dataset('info', data=ifpt_list)
#     hf.create_dataset('data', data=cc_list)
#     return 

def input_to_pandas(index, fpt_list, cc_list):
    
    labels = []
    labels = ['frame', 'mics', 'tau']
    for i in range(0,len(cc_list[0])):
        labels.append('cc_{}'.format(i))
        
    data = np.concatenate((fpt_list, cc_list), axis=1)
    df = pd.DataFrame(data, columns=labels)
    df[['frame', 'mics']] = df[['frame', 'mics']].astype(int)
    df['tau'] = df['tau'].astype(np.float16)
    df.insert(loc=0, column='index', value=f'{index:0>4}')
    return df


def save_to_csv(df_inp, df_out, index):
    import os
    inp = '{}/inp_{:0>4}.csv'.format(self.path_out, index) 
    if not os.path.isfile(inp):
        df_inp.to_csv(inp)
    out = '{}/out_{:0>4}.csv'.format(self.path_out, index)
    if not os.path.isfile(out):
        df_out.to_csv(out)
    return

def create_csvs(params):
    if params['test']:
        base_path = f"simulations/{params['mic_array']['name']}/" \
            f"test/{params['speakers']['count']}_src/"
        data_path = f"data/{params['mic_array']['name']}/" \
            f"test/{params['speakers']['count']}_src/"
    else:
        base_path = f"simulations/{params['mic_array']['name']}/" \
            f"{params['speakers']['count']}_src/"
        data_path = f"data/{params['mic_array']['name']}/" \
            f"{params['speakers']['count']}_src/"

    if params['noises']['count'] == 0:
        f_name = "clean"
    else:
        f_name = "noise"

    length = params['gcc_phat']['length'] 
    max_tau = params['gcc_phat']['max_tau'] 
    interp = params['gcc_phat']['interp']
    mic_pairs = [[1,4], [2,5], [3,6], [4,7], [5,1], [6,2], [7,3]]

    simulation_df = pd.read_csv(f'{base_path}positions_{f_name}.csv')
    label_gen = LabelGenerator(simulation_df, params)
    # hf = h5py.File(f'{data_path}gcc_{f_name}.h5', 'w')

    for i in range(1,len(simulation_df.index)+1):
        wav_file = f'{f_name}_{i:0>4}.wav'
        wav = wavfile.read(f'{base_path}{wav_file}')[1]
        logs.log_simulation(wav_file, params)

        fpt_list, cc_list = gcc_phat.prepare_input(
            wav, mic_pairs, length, max_tau, interp)
        csv_gcc = f'{data_path}gcc_{f_name}_{i:0>4}.csv'
        
        df_gcc = input_to_pandas(i, fpt_list, cc_list)
        df_gcc.to_csv(csv_gcc)
        
        n_frames = int(len(fpt_list)/len(mic_pairs)) 
        df_out = label_gen.prepare_labels(i, n_frames)
        csv_out = f'{data_path}out_{f_name}_{i:0>4}.csv'
        df_out.to_csv(csv_out)
        # df_out = output_to_pandas(FA,AP)

        # save_to_csv(df_inp, df_out, index)

    return


def main(self, noise, radius):


    self.df_sim = pd.read_csv('{}/positions.csv'.format(self.path_in))

    
    if not os.path.exists(self.path_out):
        print('Creating directory {}'.format(self.path_out))
        os.makedirs(self.path_out)


    create_csvs()

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


    create_csvs(params)


    # simulate_speech(params)
    print('DONE')
# %%