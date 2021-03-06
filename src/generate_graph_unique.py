"""
Graph generation by rnn (load model).

Use networkx library.
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json

import glob

glob.set_dir()
TYPE = 'RANDOM'

n_timestamps = 198
temperature = 0.9
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2

# read model and weights
model_file = '{}{}_{}_{}_model.json' \
             .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)
weights_file = '{}{}_{}_{}_weights.h5' \
               .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)

with open(model_file) as f:
    model = model_from_json(f.read())
model.load_weights(weights_file)
model.compile(optimizer='rmsprop', loss='mse')

# init sequences ndarray, 3-d because of training data
sentences = np.zeros((1, n_timestamps+1, NUM_OF_NODE))
sentences[:, 0, :] = 0
# init adjacency matrix
graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
edge_count = 0

# Start sampling sequences. At each iteration i the probability over
# the i-th character of each sequences is computed.
for i in np.arange(n_timestamps):
    if i % 10 == 0:
        print 'predict the {} edge'.format(i)
    probs = model.predict_proba(sentences, verbose=0)[:, i, :]
    probs = probs / np.sum(probs)
    # set temperature to control the tradeoff
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    # Go over each sequence and sample the i-th character.
    for j in np.arange(len(sentences)):
        # avoid duplicate edges explicitly
        while True:
            select_nodes = np.random.choice(NUM_OF_NODE, 2, replace=False,
                                            p=probs[j, :])
            if graph_matrix[select_nodes[0], select_nodes[1]] != 1:
                break
        for node in select_nodes:
            sentences[j, i+1, node] = 1
        # update adjacency matrix
        graph_matrix[select_nodes[0], select_nodes[1]] = 1
        graph_matrix[select_nodes[1], select_nodes[0]] = 1

# construct graph
g = nx.Graph(graph_matrix)
print nx.degree_histogram(g)
print 'num of edges:', len(g.edges())

# plot graph figure
nx.draw_spring(g)
# plt.savefig("test.png")
plt.show()

# with open(RESULT_PATH+TYPE+'_test.txt', 'w') as f:
#     f.write(sentences)
