import pandas as pd
import os

class Text():
    def __init__(self, df):
        self.df = df

def append_df_to_csv(df, csvFilePath, sep=","):

    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep,
            float_format='%.4f')
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False,
            float_format='%.4f')
    return

def read_sim_csv(file):
    '''Read the csv of the simulation to gather info of the wav files'''
    df_sim = pd.read_csv(file)
    return df_sim


if __name__ == "__main__" :
    pass