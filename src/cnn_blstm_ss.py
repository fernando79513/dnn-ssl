from numpy.lib.function_base import angle, average
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten,
    BatchNormalization, Activation, Conv2D, LSTM, Bidirectional,
    TimeDistributed, GlobalAveragePooling1D, Reshape)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
import tensorflow.keras.backend as K

from keras.utils.vis_utils import plot_model

# from keras.utils.vis_utils import plot_model
from skimage.measure import block_reduce
from tensorflow.python.keras.backend import dropout
from tensorflow.python.ops.gen_array_ops import reshape

from src.utils.emd import earth_mover_distance

# import wandb
# from wandb.keras import WandbCallback


# T = 10     T = 20
# M = 8      M = 8
# F = 1024   F = 510
# Overlap - yes

# n_saples = 512
# hop = 160
# Overlap = 60%

# T(n frames) = 20
# F(n freqs) = 257
# M = 8

# γ = 10
# Q = 2 × [360/γ] = 72

# CNN   -> 4 × 1 ; 4
# CNN   -> 3 × 3 ; 16
# CNN   -> 3 × 3 ; 32
# Dense -> 72


def locnet_cnn (input_shape=(8,257,1), output_size=72,
    filters=[4,16,32], kernels=[[4,1], [3,3], [3,3]],
    drop = 0.5):

    # I have to padd layers only verticaly
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(filters[0], (kernels[0]), activation='relu',
        padding="valid")(inputs)
    dropout1 = Dropout(drop)(conv1)
    conv2 = Conv2D(filters[1], (kernels[1]), activation='relu',
        padding="valid")(dropout1)
    dropout2 = Dropout(drop)(conv2)
    conv3 = Conv2D(filters[2], (kernels[2]), activation='relu',
        padding="valid")(dropout2)
    dropout3 = Dropout(drop)(conv3)
    flatten = Flatten(name="Flatten")(dropout3)
    outputs = Dense(output_size, activation="sigmoid", name="Output")(flatten)   
    cnn = Model(inputs=inputs,outputs=outputs, name="cnn")
    return cnn

def locnet_blstm (locnet_cnn, input_shape=(20, 8, 257), q=72, output_size=36,
    drop = 0.4):

    # TODO: use BLSTMP
    inputs = Input(shape=input_shape)
    # reshape = Reshape((20, 8, 257,1))(inputs)
    # creating BLSTM
    time_dist = TimeDistributed(locnet_cnn)(inputs)
    blstm = Bidirectional(LSTM(q, return_sequences=True, dropout=drop),
        merge_mode='ave')(time_dist)
    average = GlobalAveragePooling1D()(blstm)
    outputs = Dense(output_size, activation="sigmoid", name="Output")(average)   
    model = Model(inputs=inputs,outputs=outputs)
    return model

# def timedistr():
#     inputs = tf.keras.Input(shape=(10, 128, 128, 3))
#     conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
#     outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
#     model = Model(inputs=inputs,outputs=outputs)
#     return model

def encode_angles(angles_list, gamma):
    n_labels = 360//gamma
    labels = np.zeros((angles_list.shape[0], n_labels))

    for i, angles in enumerate(angles_list):
        for angle in angles:
            if angle != -1:
                angle_i = angle // gamma
                labels[i, angle_i] = 1
    return labels
            

def change_out_res(labels, gamma):
    output_size = labels.shape[1] // gamma
    print(output_size)
    labels = block_reduce(labels, (1, gamma), np.max)
    return labels, output_size

# def get_phasemap(stft_list):
#     n_chunks = len(stft_list)

#     output_size = labels.shape[1] // gamma
#     print(output_size)
#     labels = block_reduce(labels, (1, gamma), np.max)
#     return labels, output_size

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

    cnn = locnet_cnn()
    graph = plot_model(cnn, to_file='img/locnet_cnn.png', 
        show_shapes=True, show_dtype=True )
    blstm = locnet_blstm(cnn)
    graph = plot_model(blstm, to_file='img/locnet_blstm.png', 
        show_shapes=True, show_dtype=True )
    cnn.summary()
    blstm.summary()


    # custom_loss = earth_mover_distance()

    blstm.compile(optimizer =tf.keras.optimizers.Adam(),
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    blstm.save("data/matrix_voice/models/blstm.h5", )
    print("Saved model to disk")
    # Defining inputs and labels
    gamma = 10
    
    pmaps = np.load('data/matrix_voice/pmap_1_src_clean.npy')
    stft_data = np.load('data/matrix_voice/stft_data_1_src_clean.npy')

    # ones_src_df = df.loc[df['number of speakers'] == 1]
    # ones_src_clean_df = ones_src_df.loc[ones_src_df['number of noises'] == 0]
    # print(ones_src_df)

    # stft_np = np.array(stft_list)
    # print(stft_np.shape)
    # plt.plot(stft_np[0,0,:,0])
    # plt.show()
    # print(stft_np.shape)
    # train_inputs = np.angle(stft_np)/(2*np.pi) + .5
   
    pmaps = np.moveaxis(pmaps, [0,1,2,3], [0,-2,-1,-3])
    # plt.plot(pmaps[0,0,0,:])
    # plt.show()

    angles = stft_data[:, -2:]
    # angles = angles_df.to_numpy(dtype=np.int)
    train_labels = encode_angles(angles, gamma)
    print(train_labels[:5])

    # plt.show()
    earlyStopping = EarlyStopping(monitor="val_accuracy", 
                                patience=10,
                                verbose=1)

    checkpoint_filepath = 'data/matrix_voice/checkpoint/'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    history = History()

    history = blstm.fit(pmaps, 
                        train_labels, 
                        batch_size=128,
                        epochs=100,
                        shuffle=True, 
                        callbacks=[earlyStopping, checkpoint],
                        # callbacks=[WandbCallback(), earlyStopping],
                        validation_split=.1
                        )

