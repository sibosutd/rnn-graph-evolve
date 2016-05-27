"""
Generate degree of all indexed nodes.

Use networkx library.
"""

import sys
import networkx as nx
import numpy as np

import glob

glob.set_dir()
TYPE = 'BA_rnn'

adj_matrices = np.load(glob.DATASET_PATH+TYPE+'_adj.npy')
n_sample, n_node, n_node = adj_matrices.shape

degree_matrices = np.zeros((n_sample, n_node))

for sample in np.arange(n_sample):
    G = nx.Graph(adj_matrices[sample, :, :])
    degree_matrices[sample, :] = nx.degree(G).values()

# save stats as .npy file
np.save(glob.RESULT_PATH+TYPE+'_degree.npy', degree_matrices)
