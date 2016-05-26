"""
Train RNN with graph data.

Model: BA and RANDOM
"""

import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM

import glob

glob.set_dir()
TYPE = 'BA'

sequences = np.load(glob.DATASET_PATH+TYPE+'.npy')

NUM_OF_NODE = sequences.shape[2]
n_timestamps = sequences.shape[1]
HIDDEN_UNITS = 128
NUM_LAYER = 3

y = np.roll(sequences, -1, axis=1)

# Build the model.
model = Sequential()
model.add(LSTM(output_dim=HIDDEN_UNITS, input_dim=NUM_OF_NODE,
          return_sequences=True))
model.add(LSTM(HIDDEN_UNITS, return_sequences=True))
model.add(LSTM(HIDDEN_UNITS, return_sequences=True))
model.add(TimeDistributed(Dense(NUM_OF_NODE, activation='softmax')))
# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(sequences, y, batch_size=64, nb_epoch=50)

with open(glob.RESULT_PATH+TYPE+'_'+str(HIDDEN_UNITS)+'_'
          + str(NUM_LAYER)+'_model.json', 'w') as f:
    f.write(model.to_json())

model.save_weights(glob.RESULT_PATH+TYPE+'_'+str(HIDDEN_UNITS)+'_'
                   + str(NUM_LAYER)+'_weights.h5')

print TYPE+'_'+str(HIDDEN_UNITS)+'_'+str(NUM_LAYER)
