from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from tensorflow.keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger

import models

tf.random.set_seed(42)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-name', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')

import keras.backend as K

def loss_fn(y_true, y_pred):
        """
        categorical cross entropy (as the proposed article suggests)
                1     N    |S|            1
                - * Sigma Sigma yk log2 -----
                n    n=1   k=1          yk_hat
        S - alphabet size
        N - sequence length
        when the prediction is 0, log2(inf) = 0
        when the prediction is 1, log2(1) = 1
        """
        return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_single_output_data(file_path,batch_size,time_steps):
        print(file_path)
        series = np.load(file_path)
        series = series.reshape(-1, 1) #2d array where each row represents a dat apoint and there is only one feature per data point
        onehot_encoder = OneHotEncoder(sparse_output=False)  #one-hot encoder object that will return a dense array
        onehot_encoded = onehot_encoder.fit(series)

        series = series.reshape(-1)

        #this function creates a sliding window of size time_steps+1 
        #over the data, with a stride of 1. 
        #This means it creates sequences of length time_steps+1 
        #where the last element is used as the target output for prediction
        data = strided_app(series, time_steps+1, 1)
        l = int(len(data)/batch_size) * batch_size

        data = data[:l] 
        #extracts the input sequences (X) from the generated sliding 
        #window data. It includes all elements except the last one.
        X = data[:, :-1]
        #same as above
        Y = data[:, -1:]
        
        #his transforms the target output sequences (Y) into one-hot 
        #encoded format using the one-hot encoder fitted earlier. 
        #It encodes each integer label into a one-hot vector.
        Y = onehot_encoder.transform(Y)
        return X,Y

        
def fit_model(X, Y, bs, nb_epoch, model):
        y = Y
        optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
        model.compile(loss=loss_fn, optimizer=optim)
        checkpoint = ModelCheckpoint(arguments.name, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

        callbacks_list = [checkpoint, csv_logger, early_stopping]
        #callbacks_list = [checkpoint, csv_logger]
        model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1, shuffle=True, callbacks=callbacks_list)
 

                
                
arguments = parser.parse_args()
print(arguments)

batch_size=128
sequence_length=64
num_epochs=20

X,Y = generate_single_output_data(arguments.data,batch_size, sequence_length)
print(Y.shape[1])
model = getattr(models, arguments.model_name)(batch_size, sequence_length, Y.shape[1])
fit_model(X, Y, batch_size,num_epochs , model)

