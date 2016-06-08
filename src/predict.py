"""
Predict future links with history.

Model: BA and RANDOM
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json

import glob

glob.set_dir()
TYPE = 'BA'

n_timestamps = 198
n_t_train = 150
# n_t_test = n_timestamps - n_t_train
temperature = 1.0
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2
STOP = 150

# read data
sequences = np.load(glob.DATASET_PATH+TYPE+'.npy')

# read model and weights
with open(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)+'_'
          + str(NUM_LAYER)+'_model.json') as f:
    model = model_from_json(f.read())

model.load_weights(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)
                   + '_' + str(NUM_LAYER)+'_weights.h5')

model.compile(optimizer='rmsprop', loss='mse')

train_data = sequences[:1, 0:n_t_train, :]
y = np.roll(train_data, -1, axis=1)

model.fit(train_data, y, batch_size=64, nb_epoch=1)

# init sequences ndarray, 3-d because of training data
sentences = np.zeros((1, n_timestamps+1, NUM_OF_NODE))
sentences[:, 0:n_t_train, :] = train_data
# init adjacency matrix
graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
edge_count = 0

# Start sampling sequences. At each iteration i the probability over
# the i-th character of each sequences is computed.

for i in np.arange(1, n_t_train):
    two_node = sequences[0, i, :].nonzero()[0]
    print two_node
    graph_matrix[two_node[0], two_node[1]] = 1
    graph_matrix[two_node[1], two_node[0]] = 1

for i in np.arange(n_t_train, n_timestamps):
    if i % 10 == 0:
        print 'predict the '+str(i)+' edges...'
    probs = model.predict_proba(sentences, verbose=0)[:, i, :]
    probs = probs / np.sum(probs)
    # set temperature to control the trade-off
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    # stop intermediately
    # if i == STOP:
    #     vis_proba = probs
    #     vis_graph = graph_matrix
    #     break
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

# plot graph figure
pos = nx.nx_pydot.graphviz_layout(g)
# color = [10*p for p in vis_proba[0].tolist()]

nx.draw(g, pos, node_size=180)
# save as png
# plt.savefig(glob.RESULT_PATH+TYPE+'_'+str(STOP)+'_prob.png')
plt.show()  # display
