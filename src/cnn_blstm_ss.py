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
    outputs = Dense(output_size, activation="relu", name="Output")(flatten)   
    cnn = Model(inputs=inputs,outputs=outputs, name="cnn")
    return cnn

def locnet_blstm (locnet_cnn, input_shape=(20, 8, 257), q=72, output_size=36,
    drop = 0.4):

    # TODO: use BLSTMP
    inputs = Input(shape=input_shape)
    # creating BLSTM
    time_dist = TimeDistributed(locnet_cnn)(inputs)
    blstm = Bidirectional(LSTM(q, return_sequences=True, dropout=drop),
        merge_mode='ave')(time_dist)
    average = GlobalAveragePooling1D()(blstm)
    outputs = Dense(output_size, activation="softmax", name="Output")(average)   
    model = Model(inputs=inputs,outputs=outputs)
    return model

def locnet_blstm_ss (locnet_cnn, input_shape=(20, 8, 257), q=36,
    drop = 0.4):

    # TODO: use BLSTMP
    inputs = Input(shape=input_shape)
    # creating BLSTM
    time_dist = TimeDistributed(locnet_cnn)(inputs)
    loc_net_mask_1 = Bidirectional(LSTM(2*q, return_sequences=True, dropout=drop,
        activation='sigmoid'), merge_mode='ave',)(time_dist)
    loc_net_mask_2 = Bidirectional(LSTM(2*q, return_sequences=True, dropout=drop,
        activation='sigmoid'),merge_mode='ave')(time_dist)   
    weighted1 = Lambda(weighted_average)([loc_net_mask_1, time_dist])
    weighted2 = Lambda(weighted_average)([loc_net_mask_2, time_dist])
    output_1 = Dense(q, activation="softmax", name="Output_1")(weighted1)   
    output_2 = Dense(q, activation="softmax", name="Output_2")(weighted2)   
    concat = tf.keras.layers.Concatenate()([output_1, output_2])
    model = Model(inputs=inputs,outputs=concat)
    return model

def weighted_average(tensors):
    w = tensors[0]
    z = tensors[1]

    return tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.multiply(w, z), axis=1),
        tf.math.reduce_sum(w, axis=1))


def change_out_res(labels, gamma):
    output_size = labels.shape[1] // gamma
    print(output_size)
    labels = block_reduce(labels, (1, gamma), np.max)
    return labels, output_size

def encode_angles(angles_list, gamma):
    n_labels = 360//gamma
    labels = np.zeros((angles_list.shape[0], n_labels))

    for i, angles in enumerate(angles_list):
        if np.isscalar(angles):
            angle_i = angle // gamma
            labels[i, angle_i] = 1
        else:
            for angle in angles:
                if angle != -1:
                    angle_i = angle // gamma
                    labels[i, angle_i] = 1
    return labels

def soft_encode_angles(angles_list, gamma):
    n_labels = 360//gamma
    labels = np.zeros((angles_list.shape[0], n_labels))

    anlge_prob =  np.zeros(n_labels)
    # phi_1 = 4//gamma
    # phi_2 = 8//gamma
    phi_1 = 1
    phi_2 = 2
    for i in range(1, phi_1+1):
        anlge_prob[i] = .2
        anlge_prob[-i] = .2
    for i in range(phi_1+1, phi_2+1):
        anlge_prob[i] = .1
        anlge_prob[-i] = .1
    anlge_prob[0] = .4

    for i, angles in enumerate(angles_list):
        if np.isscalar(angles):
            angles = [angles]
        for angle in angles:
            if angle != -1:
                angle_i = angle // gamma
                labels[i] += np.roll(anlge_prob, angle_i)
        labels[i][labels[i]>=1] = 1
    return labels

def soft_encode_2_angles(angles_list, gamma):
    n_labels = 360//gamma
    labels = np.zeros((angles_list.shape[0], n_labels*2))
    labels[:,:n_labels] = soft_encode_angles(angles_list[:,0], gamma)
    labels[:,n_labels:] = soft_encode_angles(angles_list[:,1], gamma)
    return labels       


def angle_acc(y_true, y_pred):
    mse = tf.keras.metrics.mean_squared_error
    true_1, true_2 =  tf.split(y_true, num_or_size_splits=2, axis=1)
    pred_1, pred_2 =  tf.split(y_pred, num_or_size_splits=2, axis=1)
    acc_1 = tf.math.add(mse(true_1, pred_1), mse(true_2, pred_2))
    acc_2 = tf.math.add(mse(true_1, pred_2), mse(true_2, pred_1))
    acc = tf.math.minimum(acc_1, acc_2)
    return acc

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

    # Defining inputs and labels
    gamma = 10
    q = 360//gamma
    print("q is ", q)
    cnn = locnet_cnn(output_size=2*q)
    blstm = locnet_blstm_ss(cnn, q=q)
    cnn.summary()
    blstm.summary()
    # graph = plot_model(cnn, to_file='img/locnet_cnn.png', 
    #     show_shapes=True, show_dtype=True )
    graph = plot_model(blstm, to_file='img/locnet_blstm_ss.png', 
        show_shapes=True, show_dtype=True )

    custom_loss = pit_cce()
    # custom_loss = earth_mover_distance(2)
    # custom_loss = tf.keras.losses.BinaryCrossentropy()
    # custom_loss = tf.keras.losses.CategoricalCrossentropy()

    blstm.compile(optimizer =tf.keras.optimizers.Adam(),
                loss = custom_loss,
                metrics=[angle_acc])

    pmaps = np.load('data/matrix_voice/test/pmap_2_src_clean.npy')
    stft_data = np.load('data/matrix_voice/test/stft_data_2_src_clean.npy')
    print(pmaps.shape)
    print(stft_data.shape)

    print(stft_data.shape)

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
    # train_labels = encode_angles(angles, gamma)
    train_labels = soft_encode_2_angles(angles, gamma)
    print(train_labels[:20])


    blstm.save("data/matrix_voice/models/blstm_semd.h5", overwrite=True)
    print("Saved model to disk")


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

