"""
Graph generation by rnn (load model).

Adapted from char_rnn.py code
"""

import numpy as np
from keras.models import model_from_json

import glob

glob.set_dir()
TYPE = 'BA'

n_timestamps = 198
temperature = 0.8
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
    graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
    for i in np.arange(n_timestamps):
        if i % 10 == 0:
            print '%d iteration, predict the %d edges' % (j, i)
        probs = model.predict_proba(sentences, verbose=0)[j, i, :]
        probs = probs / np.sum(probs)
        # set temperature to control the tradeoff
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        while True:
            select_nodes = np.random.choice(NUM_OF_NODE, 2, replace=False,
                                            p=probs)
            if graph_matrix[select_nodes[0], select_nodes[1]] != 1:
                break

        for node in select_nodes:
            sentences[j, i+1, node] = 1
        graph_matrix[select_nodes[0], select_nodes[1]] = 1
        graph_matrix[select_nodes[1], select_nodes[0]] = 1

np.save(glob.DATASET_PATH+TYPE+'_rnn_t'+str(temperature)
        + '_uniq.npy', sentences)
