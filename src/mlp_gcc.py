from numpy.core.records import fromstring
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten,
    BatchNormalization, Activation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, History
import tensorflow.keras.backend as K

# from keras.utils.vis_utils import plot_model
from skimage.measure import block_reduce

from src.utils.emd import earth_mover_distance

# import wandb
# from wandb.keras import WandbCallback

def plot_training_history(history):
    plt.figure(figsize=(10,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy and loss')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy','Validation Accuracy', 'Loss',
                'Validation Loss'], loc='upper right')
    plt.show()

def define_model (input_shape=(7,73), output_size=360,
    hidden=[500, 450, 400], drop=0.125, l2_param=.1):

    # # Input layer
    # inputs = Input(shape=input_shape,name="Input")
    # flatten = Flatten(name="Flatten")(inputs)

    # # First layer
    # hidden1 = Dense(hidden[0],name="Hidden1")(flatten)
    # act1 = Activation("relu",name="Activation1")(hidden1)
    # drop1 = Dropout(drop, name="Dropout1")(act1)

    # # Second hidden layer
    # hidden2 = Dense(hidden[1],name="Hidden2")(drop1)
    # act2 = Activation("relu",name="Activation2")(hidden2)
    # drop2 = Dropout(drop, name="Dropout2")(act2)

    # # Third hidden layer
    # hidden3 = Dense(hidden[2],name="Hidden3")(drop2)
    # act3 = Activation("relu",name="Activation3")(hidden3)
    # drop3 = Dropout(drop, name="Dropout3")(act3)
    
    # # Output layer
    # outputs = Dense(output_size, activation="sigmoid",
    #     name="Output")(drop3)
    # # outputs = Dense(output_size, 
    # #     name="Output")(drop4)
        
    # Input layer
    inputs = Input(shape=input_shape,name="Input")
    flatten = Flatten(name="Flatten")(inputs)

    # First layer
    hidden1 = Dense(hidden[0],name="Hidden1", 
                    kernel_regularizer=l2(l2_param))(flatten)
    batch1 = BatchNormalization(name="BatchNorm1")(hidden1)
    act1 = Activation("relu",name="Activation1")(hidden1)
    drop1 = Dropout(drop, name="Dropout1")(act1)

    # Second hidden layer
    hidden2 = Dense(hidden[1],name="Hidden2", 
                    kernel_regularizer=l2(l2_param))(drop1)
    batch2 = BatchNormalization(name="BatchNorm2")(hidden2)
    act2 = Activation("relu",name="Activation2")(hidden2)
    drop2 = Dropout(drop, name="Dropout2")(act2)

    # Third hidden layer
    hidden3 = Dense(hidden[2],name="Hidden3", 
                    kernel_regularizer=l2(l2_param))(drop2)
    batch3 = BatchNormalization(name="BatchNorm3")(hidden3)
    act3 = Activation("relu",name="Activation3")(batch3)
    drop3 = Dropout(drop, name="Dropout3")(act3)
    
    # Output layer
    outputs = Dense(output_size, activation="sigmoid",
        name="Output")(drop3)
    # outputs = Dense(output_size, 
    #     name="Output")(drop4)
        
    model = Model(inputs=inputs, outputs=outputs)
    return model

def change_out_res(labels, gamma):
    output_size = labels.shape[1] // gamma
    print(output_size)
    labels = block_reduce(labels, (1, gamma), np.max)
    return labels, output_size

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

    mic_pairs = params['gcc_phat']['mic_pairs']
    
    gcc_df = pd.read_feather('data/matrix_voice/gcc.ftr')
    out_df = pd.read_feather('data/matrix_voice/out.ftr')
    ones_src_df = out_df.loc[out_df['number of speakers'] == 1]
    ones_src_clean_df = ones_src_df.loc[ones_src_df['number of noises'] == 0]


    angles_df = out_df.filter(regex='^[0-9]*$', axis=1)
    output_size = len(angles_df.columns)
    train_labels = np.array(angles_df)

    # thresh = 1
    # train_labels[train_labels >= thresh] = 1
    # train_labels[train_labels < thresh] = 0
    
    train_labels, output_size = change_out_res(train_labels, 10)


    cc_df = gcc_df.filter(regex='^cc_[0-9]*$', axis=1)

    train_inputs = np.array(cc_df)
    train_inputs = np.reshape(train_inputs, (
        -1, len(mic_pairs), len(cc_df.columns)))
    input_shape = (len(mic_pairs),len(cc_df.columns))

    train_inputs = train_inputs * 2 + .5

    print(train_inputs.shape)
    # train_labels = train_labels.reshape((-1,1,360))
    # print(train_labels.shape)


    # run = wandb.init()
    # config = run.config
    # config.epochs = 10


    model = define_model(input_shape, output_size=output_size)
    # plot_model(model, to_file='img/model.png')

    # def custom_loss(y_true, y_pred):
    #     return K.mean(y_true - y_pred)**2
    # custom_loss = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    # )
    custom_loss = earth_mover_distance()

    model.compile(optimizer =tf.keras.optimizers.Adam(),
                loss = custom_loss,
                metrics=['accuracy'])

    # plt.plot(train_inputs[7900].T)
    # plt.show()
    # plt.plot(train_labels[7900])
    # plt.show()
    earlyStopping = EarlyStopping(monitor="val_accuracy", 
                                patience=10,
                                verbose=1,
                                restore_best_weights=True)
    history = History()

    history = model.fit(train_inputs, 
                        train_labels, 
                        # batch_size=512,
                        epochs=100,
                        shuffle=True, 
                        callbacks=[earlyStopping],
                        # callbacks=[WandbCallback(), earlyStopping],
                        validation_split=.2
                        )
    model.save("data/model.h5", )
    print("Saved model to disk")