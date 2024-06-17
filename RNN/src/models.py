from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, GRU
from math import sqrt
from tensorflow.keras.layers import Embedding, Input, Dense, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
from keras import backend as K
from tensorflow.keras.models import Model

# coding=utf-8

import numpy as np
import torch
import time


def biLSTM(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biGRU(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

#def biLSTM(bs,time_steps,alphabet_size):
#        model = Sequential()
#        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
#        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=True)))
#        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
#        model.add(Dense(64, activation='relu'))
#        model.add(Dense(alphabet_size, activation='softmax'))
#        return model