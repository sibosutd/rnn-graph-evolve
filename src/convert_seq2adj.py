"""
Convert graph sequences to adjacency matrix.

Use networkx library.
"""

import sys
import networkx as nx
import numpy as np

import glob

glob.set_dir()
TYPE = 'BA'

sequences = np.load(glob.DATASET_PATH+TYPE+'.npy')
n_sample, n_timestamps, n_node = sequences.shape

matrices = np.zeros((n_sample, n_node, n_node))

for sample in np.arange(n_sample):
    print sample
    D = nx.Graph(np.zeros((n_node, n_node)))
    for step in np.arange(1, n_timestamps):
        timestamp = sequences[sample, step, :]
        indices = timestamp.nonzero()
        D.add_edges_from(tuple(indices))
    matrices[sample, :, :] = nx.to_numpy_matrix(D)

# save stats as .npy file
np.save(glob.DATASET_PATH+TYPE+'_adj.npy', matrices)
