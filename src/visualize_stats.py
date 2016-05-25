"""
Visualize statistics of single graph.

Use networkx library.
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = '../'
DATASET_PATH = BASE_PATH + 'datasets/'
RESULT_PATH = BASE_PATH + 'results/'
TYPE = 'BA'

sequences = np.load(DATASET_PATH+TYPE+'.npy')
n_sample, n_timestamps, n_node = sequences.shape

diam_mat = np.zeros((n_sample, n_timestamps))

num_select = 5000
for sample in np.arange(num_select):
    print sample
    D = nx.Graph(np.zeros((n_node, n_node)))
    for step in np.arange(1, n_timestamps):
        timestamp = sequences[sample, step, :]
        indices = timestamp.nonzero()
        D.add_edges_from(tuple(indices))
        # generate largest connected component
        largest_cc = max(nx.connected_component_subgraphs(D), key=len)
        diam_mat[sample, step] = nx.diameter(largest_cc)

curve = np.zeros((n_timestamps, 3))
curve[:, 1] = np.mean(diam_mat[:num_select, :], axis=0)
curve[:, 0] = np.mean(diam_mat[:num_select, :], axis=0)-np.std(diam_mat[:num_select, :], axis=0)
curve[:, 2] = np.mean(diam_mat[:num_select, :], axis=0)+np.std(diam_mat[:num_select, :], axis=0)

plt.plot(np.arange(n_timestamps), curve[:, 1], 'k', linewidth=2)
plt.fill_between(np.arange(n_timestamps), curve[:, 0], curve[:, 2],
                 alpha=.25, linewidth=0)
plt.grid(True)
# nx.draw(D)
# plt.savefig("test.png")
plt.show()

# print nx.diameter(D)
