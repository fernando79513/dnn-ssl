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


import visualkeras



from src.combine_data import shuffle_df

config_file = 'config/param_matrix_voice.json'
with open(config_file) as f:
    params = json.load(f)
mic_pairs = params['gcc_phat']['mic_pairs']



model = tf.keras.models.load_model('data/matrix_voice/models/blstm_pit.h5',
    custom_objects={
        "_pit_earth_mover_distance": pit_earth_mover_distance,
        "_pit_cce": pit_cce,
        "ape": ape,
        "_roll_earth_mover_distance":roll_earth_mover_distance,
        })
print('model loaded')
print('im here')
visualkeras.layered_view(model).show() 
