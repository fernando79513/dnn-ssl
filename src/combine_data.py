import numpy as np
import pandas as pd
import os
import json
import os

def shuffle_df(df_1, df_2, n=1):
    '''Shuffle 2 dataframes where the first has n times more lines than
    the second'''

    if len(df_2)*n != len(df_1):
        print("The dataframes dont have matching lenght")
        return df_1, df_2

    ind_2 = np.array(df_2.index)
    np.random.shuffle(ind_2)

    for i in range(n):
        if i == 0:
            ind_1 = ind_2*n
        else:
            ind_1 = np.vstack((ind_1, ind_2*n+i))
    ind_1 = ind_1.T.flatten()

    shuffled_df_1 = df_1.loc[ind_1, :].reset_index()
    shuffled_df_1['index'] = shuffled_df_1['index'].apply(lambda x : x // n)
    shuffled_df_2 = df_2.loc[ind_2, :].reset_index()
    return shuffled_df_1, shuffled_df_2

def combine_csvs(params):

    mic_pairs = params['gcc_phat']['mic_pairs']

    if params['test']:
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


    # if not params['test']:
    if params['test']:
        gcc_df.to_csv(f'{path}gcc.csv', index=False)
        out_df.to_csv(f'{path}out.csv', index=False)
    # else:
    #     gcc_df, out_df = shuffle_df(gcc_df, out_df, len(mic_pairs))

    gcc_df.to_feather(f'{path}gcc.ftr')
    out_df.to_feather(f'{path}out.ftr')

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

    mic_pairs = params['gcc_phat']['mic_pairs']

    combine_csvs(params)
