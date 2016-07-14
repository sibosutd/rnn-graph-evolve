"""
Train RNN with real-world graph data.

Model: BA and RANDOM
"""

import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM

import glob

glob.set_dir()

sequences = np.load(glob.DATASET_PATH+'real_seq.npy')

NUM_OF_NODE = sequences.shape[2]
n_timestamps = sequences.shape[1]
HIDDEN_UNITS = 64
NUM_LAYER = 1

y = np.roll(sequences, -1, axis=1)

# Build the model.
model = Sequential()
model.add(LSTM(output_dim=HIDDEN_UNITS, input_dim=NUM_OF_NODE,
          return_sequences=True))
# model.add(LSTM(HIDDEN_UNITS, return_sequences=True))
# model.add(LSTM(HIDDEN_UNITS, return_sequences=True))
model.add(TimeDistributed(Dense(NUM_OF_NODE, activation='softmax')))
# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(sequences, y, batch_size=64, nb_epoch=50)
