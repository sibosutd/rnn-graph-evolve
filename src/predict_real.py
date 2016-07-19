"""
Predict future links with given training data.

Model: BA and RANDOM
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, LSTM

import glob

glob.set_dir()
TYPE = 'real'

# load network data
orig_seq = np.load(glob.DATASET_PATH+TYPE+'.npy')
adj_matrices = np.load(glob.DATASET_PATH+TYPE+'_adj.npy')

_, n_timestamps, NUM_OF_NODE = orig_seq.shape
# n_timestamps = 1500
# NUM_OF_NODE = 1000
n_t_train = 1200
temperature = .9
HIDDEN_UNITS = 128
NUM_LAYER = 2
select_index = 2

# format file name of models and weights
model_file = '{}{}_{}_{}_model.json' \
             .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)
weights_file = '{}{}_{}_{}_weights.h5' \
               .format(glob.MODEL_PATH, TYPE, HIDDEN_UNITS, NUM_LAYER)

# create a stateful version model
model = Sequential()
model.add(LSTM(output_dim=HIDDEN_UNITS,
               input_dim=NUM_OF_NODE,
               batch_input_shape=(1, 1, NUM_OF_NODE),
               return_sequences=True,
               stateful=True))
model.add(LSTM(HIDDEN_UNITS,
               batch_input_shape=(1, 1, NUM_OF_NODE),
               return_sequences=True,
               stateful=True))
model.add(TimeDistributed(Dense(NUM_OF_NODE, activation='softmax')))

# load non-stateful model weight
model.load_weights(weights_file)
model.compile(optimizer='rmsprop', loss='mse')
model.reset_states()

# print parameters
print 'Parameter:\n\tSteps: {}\n\tTemp: {}'.format(n_t_train, temperature)

# init prediction sequences ndarray, 3-d because of training data
pred_seq = np.zeros((1, n_timestamps+1, NUM_OF_NODE))
pred_seq[:, :n_t_train, :] = orig_seq[[select_index], :n_t_train, :]
# init adjacency matrix
graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
graph_matrix_train = np.zeros((NUM_OF_NODE, NUM_OF_NODE))
edge_count = 0

# store training data as adjacency matrix
for i in np.arange(1, n_t_train):
    # silly way of handling dimensions
    model.fit(orig_seq[[select_index], i-1:i, :],
              orig_seq[[select_index], i:i+1, :],
              batch_size=1,
              nb_epoch=1,
              verbose=0)

    two_node = orig_seq[select_index, i, :].nonzero()[0]
    graph_matrix[two_node[0], two_node[1]] = 1
    graph_matrix[two_node[1], two_node[0]] = 1
    graph_matrix_train[two_node[0], two_node[1]] = 1
    graph_matrix_train[two_node[1], two_node[0]] = 1
    print 'train the {} edge: {}'.format(i, two_node)

# predicting stage
for i in np.arange(n_t_train, n_timestamps):
    # predict the probability of next edge
    probs = model.predict_proba(pred_seq[:, [i-1], :], verbose=0)
    probs = probs / np.sum(probs)
    # set temperature to control the trade-off
    probs = np.log(probs) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    # sampling
    for j in np.arange(len(pred_seq)):
        # avoid duplicate edges explicitly
        while True:
            # note: prob must be 1-d array
            select_nodes = np.random.choice(NUM_OF_NODE, 2,
                                            replace=False,
                                            p=probs[j, 0, :])
            if graph_matrix[select_nodes[0], select_nodes[1]] != 1:
                break
        for node in select_nodes:
            pred_seq[j, i, node] = 1
        # update adjacency matrix
        graph_matrix[select_nodes[0], select_nodes[1]] = 1
        graph_matrix[select_nodes[1], select_nodes[0]] = 1
    # dynamic evaluation using stateful rnn, update the model
    # model.fit(orig_seq[[select_index], i-1:i, :],
    #           pred_seq[:, i:i+1, :],
    # model.fit(pred_seq[:, [i-1], :],
    #           pred_seq[:, [i], :],
    #           batch_size=1,
    #           nb_epoch=1,
    #           verbose=0)
    print 'predict the {} edge: {}'.format(i, select_nodes)

# construct graphs
g_0 = nx.Graph(graph_matrix_train)
g_1 = nx.Graph(adj_matrices[select_index, :, :])
g_2 = nx.Graph(graph_matrix)

# extract degree of nodes
hist_0 = np.asarray(nx.degree(g_0).values())
hist_1 = np.asarray(nx.degree(g_1).values()) \
         - np.asarray(nx.degree(g_0).values())
hist_2 = np.asarray(nx.degree(g_2).values()) \
         - np.asarray(nx.degree(g_0).values())

# degree distribution
num_bins = 50
degree_dist_0 = np.asarray(nx.degree(g_0).values())
degree_dist_1 = np.asarray(nx.degree(g_1).values())
degree_dist_2 = np.asarray(nx.degree(g_2).values())

# plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
axes[0, 0].set_title('Degrees of nodes - Ground truth')
axes[1, 0].set_title('Degrees of nodes - Predicted')
axes[0, 0].bar(np.arange(NUM_OF_NODE), hist_0, color='k')
axes[1, 0].bar(np.arange(NUM_OF_NODE), hist_0, color='k')
axes[0, 0].bar(np.arange(NUM_OF_NODE), hist_1, bottom=hist_0, color='r')
axes[1, 0].bar(np.arange(NUM_OF_NODE), hist_2, bottom=hist_0, color='b')
axes[1, 0].set_ylim(axes[0, 0].get_ylim())

axes[0, 1].set_title('Degrees distribution - Ground truth')
axes[1, 1].set_title('Degrees distribution - Predicted')
# axes[2].bar(np.arange(100), hist_0, color='k')
# axes[3].bar(np.arange(100), hist_0, color='k')
axes[0, 1].hist(degree_dist_1, num_bins, range=(0, num_bins), color='r')
axes[1, 1].hist(degree_dist_2, num_bins, range=(0, num_bins), color='b')
axes[1, 1].set_ylim(axes[0, 1].get_ylim())

# display
plt.show()
