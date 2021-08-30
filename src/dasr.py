from numpy.lib.function_base import angle, average
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten,
    Conv2D, LSTM, Bidirectional, TimeDistributed, GlobalAveragePooling1D,
    Lambda)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
import tensorflow.keras.backend as K

from keras.utils.vis_utils import plot_model

# from keras.utils.vis_utils import plot_model
from skimage.measure import block_reduce
from tensorflow.python.keras.backend import dropout, sigmoid
from tensorflow.python.ops.gen_array_ops import reshape

from src.utils.emd import earth_mover_distance, pit_earth_mover_distance, pit_cce

# import wandb
# from wandb.keras import WandbCallback

F = 257
T = 20
T_ARRAY = np.linspace(0, 0.019, F)
N = 2
R = 0.03829
C = 343
FS = 16000
F_ARRAY = np.linspace(0,8000,F)
MIC_ANGLES = [
     3.0479777621063042,  2.1505430344911307,  1.2529265224648303,
     0.3551953279695166, -0.5422998705472181, -1.4398667411390826,
    -2.337544759805849  ]
M = len(MIC_ANGLES)


def get_steering_vector(angles):
    # todo change for tensors
    tau = np.empty((N, M))
    d   = np.empty((N, M, F))
    for n, theta in enumerate(angles):
        for m, mic_angle in enumerate(MIC_ANGLES):
            tau[n,m] = R/C * tf.math.cos(theta - mic_angle)
            for f, freq in enumerate(F_ARRAY):
                d[n,m,f] = tf.math.exp(-1j*2*np.pi*freq*tau[n,m])
    return d

def get_angle_features(d, y):
    # todo change for tensors
    a = np.empty((N, F, T))
    a_pred = np.empty((N, F, T))
    for n in range(N):
        for t, time in enumerate(T_ARRAY):
            a_pred[i,:,j] = np.abs(d_1[i].conj().T*stft_mc[i+1,:,j])
            a_2[i,:,j] = np.abs(d_2[i].conj().T*stft_mc[i+1,:,j])

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Train and test the MLP network')
    parser.add_argument('-c', '--config_file', type=str)
    args = parser.parse_args()

    if args.config_file == None:
         args.config_file = 'config/param_matrix_voice.json'
    with open(args.config_file) as f:
        params = json.load(f)
