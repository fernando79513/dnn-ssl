import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import json

# import tensorflow as tf
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
from src.utils.emd import earth_mover_distance

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

def change_out_res(labels, gamma):
    output_size = labels.shape[1] // gamma
    print(output_size)
    labels = block_reduce(labels, (1, gamma), np.max)
    return labels, output_size


if __name__ == "__main__":

    from src.combine_data import shuffle_df

    config_file = 'config/param_matrix_voice.json'
    with open(config_file) as f:
        params = json.load(f)
    mic_pairs = params['gcc_phat']['mic_pairs']

    model = keras.models.load_model('data/model.h5', custom_objects={"_earth_mover_distance": earth_mover_distance})
    print('model loaded')
    
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
        plt.show()