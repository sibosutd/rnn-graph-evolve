"""
Graph generation by rnn (load model).

Adapted from char_rnn.py code
"""

import numpy as np
from keras.models import model_from_json

import glob

glob.set_dir()
TYPE = 'RANDOM'

n_timestamps = 198
temperature = 1.0
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2

n_sample = 100

# read model and weights
with open(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)+'_'
          + str(NUM_LAYER)+'_model.json') as f:
    model = model_from_json(f.read())

model.load_weights(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)
                   + '_' + str(NUM_LAYER)+'_weights.h5')

model.compile(optimizer='rmsprop', loss='mse')

sentences = np.zeros((n_sample, n_timestamps+1, NUM_OF_NODE))
sentences[:, 0, :] = 0

# sampling two-hot vectors.
for j in np.arange(n_sample):
    for i in np.arange(n_timestamps):
        if i % 10 == 0:
            print str(j)+' iteration, predict the '+str(i)+' edges'
        probs = model.predict_proba(sentences, verbose=0)[j, i, :]
        probs = probs / np.sum(probs)
        # set temperature to control the tradeoff
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        # Go over each sequence and sample the i-th character.
        select_nodes = np.random.choice(NUM_OF_NODE, 2, replace=False,
                                        p=probs)
        for node in select_nodes:
            sentences[j, i+1, node] = 1

np.save(glob.DATASET_PATH+TYPE+'_rnn_t'+str(temperature)+'.npy', sentences)
