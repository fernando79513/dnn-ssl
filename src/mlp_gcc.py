from numpy.lib.shape_base import split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Input, Dense, GaussianNoise, Dropout, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.constraints import UnitNorm
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model


def define_model (input_shape, hidden=[1024, 512, 512],
    l2_param=.1, drop=0.125):

    # Input layer
    inputs = Input(shape=(input_shape,),name="Input")

    # First layer
    hidden1 = Dense(hidden[0],name="Hidden1",use_bias=False, 
                    kernel_regularizer=l2(l2_param))(inputs)
    batch1 = BatchNormalization(name="BatchNorm1")(hidden1)
    act1 = Activation("relu",name="Activation1")(batch1)
    drop1 = Dropout(drop, name="Dropout1")(act1)

    # Second hidden layer
    hidden2 = Dense(hidden[1],name="Hidden2",use_bias=False, 
                    kernel_regularizer=l2(l2_param))(drop1)
    batch2 = BatchNormalization(name="BatchNorm2")(hidden2)
    act2 = Activation("relu",name="Activation2")(batch2)
    drop2 = Dropout(drop, name="Dropout2")(act2)

    # Third hidden layer
    hidden3 = Dense(hidden[2],name="Hidden3",use_bias=False, 
                    kernel_regularizer=l2(l2_param))(drop2)
    batch3 = BatchNormalization(name="BatchNorm3")(hidden3)
    act3 = Activation("relu",name="Activation3")(batch3)
    drop3 = Dropout(drop, name="Dropout3")(act3)

    # Output layer
    outputs = Dense(1, activation='linear',name="Output")(drop3)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":

    # model = define_model(73*7)
    # plot_model(model, to_file='img/model.png')

    output = pd.read_feather('data/matrix_voice/test/out.ftr')
    print(output)
