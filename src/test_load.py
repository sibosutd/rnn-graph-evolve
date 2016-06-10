"""
Load non-stateful model weight to a stateful model.

Model: BA and RANDOM
"""

import sys
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM

import glob

glob.set_dir()
TYPE = 'BA'

n_timestamps = 198
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2

# read model and weights
# model_file = '{}{}_{}_{}_model.json' \
#              .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)
weights_file = '{}{}_{}_{}_weights.h5' \
               .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)

model = Sequential()
model.add(LSTM(output_dim=HIDDEN_UNITS, input_dim=NUM_OF_NODE, batch_input_shape=(1, n_timestamps, 1), return_sequences=True, stateful=True))
model.add(LSTM(HIDDEN_UNITS, batch_input_shape=(1, n_timestamps, 1), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(NUM_OF_NODE, activation='softmax')))

# with open(model_file) as f:
#     model = model_from_json(f.read())
model.load_weights(weights_file)
model.compile(optimizer='rmsprop', loss='mse')
model.summary()
