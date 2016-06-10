"""
Predict future links with given training data.

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
temperature = 1.0
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2
select_index = 1

# read data
sequences = np.load(glob.DATASET_PATH+TYPE+'.npy')
adj_matrices = np.load(glob.DATASET_PATH+TYPE+'_adj.npy')

# read model and weights
model_file = '{}{}_{}_{}_model.json' \
             .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)
weights_file = '{}{}_{}_{}_weights.h5' \
               .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)

# with open(model_file) as f:
#     model = model_from_json(f.read())
# model.load_weights(weights_file)
# model.compile(optimizer='rmsprop', loss='mse')

# init sequences ndarray, 3-d because of training data
sentences = np.zeros((1, n_timestamps+1, NUM_OF_NODE))
sentences[:, 0:n_t_train, :] = sequences[[select_index], 0:n_t_train, :]
# init adjacency matrix
graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
graph_matrix_train = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
edge_count = 0

# store training data as adjacency matrix
for i in np.arange(1, n_t_train):
    two_node = sequences[select_index, i, :].nonzero()[0]
    graph_matrix[two_node[0], two_node[1]] = 1
    graph_matrix[two_node[1], two_node[0]] = 1
    graph_matrix_train[two_node[0], two_node[1]] = 1
    graph_matrix_train[two_node[1], two_node[0]] = 1

for i in np.arange(n_t_train, n_timestamps):
    print 'predict the {} edge'.format(i)
    # dynamic evaluation using stateful rnn
    with open(model_file) as f:
        model = model_from_json(f.read())
    model.load_weights(weights_file)
    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(sequences[[select_index], :i, :],
              sentences[:, 1:i+1, :], nb_epoch=1, verbose=0)
    probs = model.predict_proba(sentences, verbose=0)[:, i, :]
    probs = probs / np.sum(probs)
    # set temperature to control the trade-off
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    # sampling
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
g_0 = nx.Graph(graph_matrix_train)
g_1 = nx.Graph(adj_matrices[select_index, :, :])
g_2 = nx.Graph(graph_matrix)

hist_0 = np.asarray(nx.degree(g_0).values())
hist_1 = np.asarray(nx.degree(g_1).values()) \
         - np.asarray(nx.degree(g_0).values())
hist_2 = np.asarray(nx.degree(g_2).values()) \
         - np.asarray(nx.degree(g_0).values())

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

axes[0].set_title('Degrees of nodes - Ground truth')
axes[1].set_title('Degrees of nodes - Predicted')

axes[0].bar(np.arange(100), hist_0, color='k')
axes[1].bar(np.arange(100), hist_0, color='k')

axes[0].bar(np.arange(100), hist_1, bottom=hist_0, color='r')
axes[1].bar(np.arange(100), hist_2, bottom=hist_0, color='b')

# plot graph figure
# pos = nx.nx_pydot.graphviz_layout(g)
# color = [10*p for p in vis_proba[0].tolist()]

# nx.draw(g, pos, node_size=180)
# save as png
# plt.savefig(glob.RESULT_PATH+TYPE+'_'+str(STOP)+'_prob.png')
plt.show()  # display
