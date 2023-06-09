from src.utils.emd import earth_mover_distance, pit_cce, pit_earth_mover_distance, roll_earth_mover_distance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import json
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
# import tensorflow as tf
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle

from src.cnn_blstm_ss import (encode_angles, soft_encode_angles, 
    soft_encode_2_angles, ape)

import wandb
from wandb.keras import WandbCallback




if __name__ == "__main__":

    from src.combine_data import shuffle_df

    config_file = 'config/param_matrix_voice.json'
    with open(config_file) as f:
        params = json.load(f)
    mic_pairs = params['gcc_phat']['mic_pairs']


    model = tf.keras.models.load_model('data/matrix_voice/models/blstm_semd.h5',
        custom_objects={
            "_pit_earth_mover_distance": pit_earth_mover_distance,
            "_pit_cce": pit_cce,
            "ape": ape,
            "_roll_earth_mover_distance":roll_earth_mover_distance,
            })
    print('model loaded')
    checkpoint_filepath = 'data/matrix_voice/checkpoint/cnn_blstm_cce_1_src/'
    model.load_weights(checkpoint_filepath)
    print('weights loaded')

    gamma = 1
    pmaps = np.load('data/matrix_voice/test/pmap_2_src_clean.npy')
    stft_data = np.load('data/matrix_voice/test/stft_data_2_src_clean.npy')
    # pmaps = np.load('data/real/pmap_real.npy')
    # stft_data = np.load('data/real/stft_data_real.npy')
    
    # for i in range (100,1000,100):
    #     new_pmaps = np.load(f'data/matrix_voice/dev/pmap_1_src_clean_{i}.npy')
    #     new_stft = np.load(f'data/matrix_voice/dev/stft_data_1_src_clean_{i}.npy')
    #     pmaps = np.concatenate((pmaps, new_pmaps))
    #     stft_data = np.concatenate((stft_data, new_stft))
    #     print(f'concatenated 1_src_clean {i}')
   
    pmaps = np.moveaxis(pmaps, [0,1,2,3], [0,-2,-1,-3])
    angles = stft_data[:, -2:]
    test_labels = soft_encode_2_angles(angles, gamma)
    print(test_labels.shape)
    # exit()
    # print(test_labels[:5])

    # s_pmaps, s_labels, angles = shuffle(pmaps, test_labels, angles)
    s_pmaps, s_labels, angles = pmaps[255:265], test_labels[255:265], angles[255:265]
    # import pdb;pdb.set_trace()


    # print("Evaluate on test data")
    # results = model.evaluate(test_inputs, test_labels, batch_size=128)
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    t = time.perf_counter()
    predictions = model.predict(s_pmaps[:10])
    print("predictions shape:", predictions.shape)
    elapsed = time.perf_counter() -t 
    print(elapsed)

    print(angles[0])
    print(s_labels[0])

    for i in range(10):
        # print(predictions[i])
        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(10, 5)
        axs.set_title('CNN-BLSTM-SS | SMED | Dropout = 0,4')
        axs.set_xlabel('Ángulo')
        axs.set_xlim(0, 360)
        axs.set_ylabel('Probabilidad')
        axs.set_ylim(0, 0.6)
        axs.grid(True)
        axs.plot(s_labels[i,:(360//gamma)]+s_labels[i,(360//gamma):])
        axs.plot(predictions[i,(360//gamma):])
        axs.plot(predictions[i,:(360//gamma)])
        # plt.plot(s_labels[i])
        # plt.plot(predictions[i])
        # plt.show()
        plt.savefig(f"img/pred_{i:0>4}")
        plt.close()


