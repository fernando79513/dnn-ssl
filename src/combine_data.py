from src.utils import gcc_phat
from src.utils.csv_utils import combine_csvs
from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py

def combine_csvs(params):
    if params['test'] :
        path =  f'data/{params["mic_array"]["name"]}/test/'
    else:
        path = f'data/{params["mic_array"]["name"]}/'

    df = pd.read_csv(f'{path}data.csv')


    gcc_list = []
    out_list = []
    for _, row in df.iterrows():
        name = row[0]
        ftype = row[1]
        n_files = row[2]
        for i in range(1, n_files + 1):
            gcc_file = f'{path}{name}/gcc_{ftype}_{i:0>4}.csv'
            if not os.path.isfile(gcc_file):
                print('No gcc file')
                continue
            out_file = f'{path}{name}/out_{ftype}_{i:0>4}.csv'
            if not os.path.isfile(out_file):
                print('No out file')
                continue
            gcc_list.append(gcc_file)
            out_list.append(out_file)

        
    gcc_df = pd.concat([pd.read_csv(f, index_col=0) for f in gcc_list], 
        ignore_index=True).fillna(0)
    out_df = pd.concat([pd.read_csv(f, index_col=0) for f in out_list],
        ignore_index=True).fillna(0)

    print(gcc_df.isnull().sum().sum())
    print(out_df.isnull().sum().sum())

    gcc_df.to_feather(f'{path}gcc.ftr')
    out_df.to_feather(f'{path}out.ftr')
    # gcc_df.to_csv(f'{path}gcc.csv')
    # out_df.to_csv(f'{path}out.csv')

    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Combine all csvs in one big file for training')
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-t', '--test', type=bool, nargs='?', const=True)
    args = parser.parse_args()

    if args.config_file == None:
         args.config_file = 'config/param_matrix_voice.json'
    with open(args.config_file) as f:
        params = json.load(f)
    if args.test != None:
        params['test'] = args.test

    combine_csvs(params)
