import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import json

# import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, 
    BatchNormalization, Activation)
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, History

from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix,classification_report
from skimage.measure import block_reduce

from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.saving import saved_model
from src.utils.emd import earth_mover_distance
from src.cnn_blstm_ss import encode_angles

import wandb
from wandb.keras import WandbCallback



def test_model(model,x,y):
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y, y_pred))

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    from src.combine_data import shuffle_df

    config_file = 'config/param_matrix_voice.json'
    with open(config_file) as f:
        params = json.load(f)
    mic_pairs = params['gcc_phat']['mic_pairs']

    model = tf.keras.models.load_model('data/matrix_voice/models/blstm.h5')
    print('model loaded')
    checkpoint_filepath = 'data/matrix_voice/checkpoint/'
    model.load_weights(checkpoint_filepath)
    print('weights loaded')

    gamma = 10
    pmaps = np.load('data/matrix_voice/test/pmap_1_src_clean.npy')
    stft_data = np.load('data/matrix_voice/stft_data_1_src_clean.npy')

   
    pmaps = np.moveaxis(pmaps, [0,1,2,3], [0,-2,-1,-3])
    angles = stft_data[:, -2:]
    train_labels = encode_angles(angles, gamma)
    print(train_labels[:5])

    
    exit()
    gcc_df = pd.read_feather('data/matrix_voice/test/gcc.ftr')
    out_df = pd.read_feather('data/matrix_voice/test/out.ftr')

    gcc_df, out_df = shuffle_df(gcc_df, out_df, len(mic_pairs))

    angles_df = out_df.filter(regex='^[0-9]*$', axis=1)
    output_size = len(angles_df.columns)
    test_labels = np.array(angles_df)
    test_labels, output_size = change_out_res(test_labels, 10)



    cc_df = gcc_df.filter(regex='^cc_[0-9]*$', axis=1)

    test_inputs = np.array(cc_df)
    test_inputs = np.reshape(test_inputs, (
        -1, len(mic_pairs), len(cc_df.columns)))
    input_shape = (len(mic_pairs),len(cc_df.columns))

    test_inputs = test_inputs * 2 + .5


    

    # print("Evaluate on test data")
    # results = model.evaluate(test_inputs, test_labels, batch_size=128)
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(test_inputs[:10])
    print("predictions shape:", predictions.shape)

    for i in range(10):
        plt.plot(test_labels[i])
        plt.plot(predictions[i])
        plt.savefig(f"img/pred_{1:0>4}")