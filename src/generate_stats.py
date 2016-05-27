"""
Generate statistics of all graphs.

Use networkx library.
"""

import sys
import networkx as nx
import numpy as np

import glob

glob.set_dir()
TYPE = 'RANDOM_rnn'

sequences = np.load(glob.DATASET_PATH+TYPE+'.npy')
n_sample, n_timestamps, n_node = sequences.shape

diam_mat = np.zeros((n_sample, n_timestamps))

for sample in np.arange(n_sample):
    print sample
    D = nx.Graph(np.zeros((n_node, n_node)))
    for step in np.arange(1, n_timestamps):
        timestamp = sequences[sample, step, :]
        indices = timestamp.nonzero()
        D.add_edges_from(tuple(indices))
        # generate largest connected subgraph
        largest_cc = max(nx.connected_component_subgraphs(D), key=len)
        diam_mat[sample, step] = nx.diameter(largest_cc)

# save stats as .npy file
np.save(glob.RESULT_PATH+TYPE+'_diam.npy', diam_mat)
