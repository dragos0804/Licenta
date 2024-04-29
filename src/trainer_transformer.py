import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.preprocessing import OneHotEncoder
import models 

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(0)

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

def generate_single_output_data(file_path, sequence_length):
    series = np.load(file_path)
    X = series[:, :-1]  # Input sequences
    Y = series[:, -1]   # Target output sequences
    return X, Y

def fit_model(X, Y, batch_size, num_epochs, model):
    optim = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=optim)

    # Define callbacks
    checkpoint = ModelCheckpoint("transformer_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("training_log.csv", append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)
    callbacks_list = [checkpoint, csv_logger, early_stopping]

    # Train the model
    model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, verbose=1, callbacks=callbacks_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', default=None, dest='data', help='choose sequence file')
    parser.add_argument('-bs', action='store', type=int, default=128, dest='batch_size', help='batch size')
    parser.add_argument('-seq_len', action='store', type=int, default=64, dest='sequence_length', help='sequence length')
    parser.add_argument('-epochs', action='store', type=int, default=20, dest='num_epochs', help='number of epochs')
    arguments = parser.parse_args()
    print(arguments)

    X, Y = generate_single_output_data(arguments.data, arguments.sequence_length)

    model = transformer(sequence_length=arguments.sequence_length, output_dim=np.max(Y)+1)
    fit_model(X, Y, arguments.batch_size, arguments.num_epochs, model)

if __name__ == "__main__":
    main()