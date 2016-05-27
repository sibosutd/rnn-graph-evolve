"""
Generate degree histogram of all graphs.

Use networkx library.
"""

import networkx as nx
import numpy as np

import glob

glob.set_dir()
TYPE = 'BA_rnn'
n_bins = 50

adj_matrices = np.load(glob.DATASET_PATH+TYPE+'_adj.npy')
n_sample, n_node, n_node = adj_matrices.shape

degree_matrices = np.zeros((n_sample, n_bins))

for sample in np.arange(n_sample):
    degree_vec = np.zeros((n_bins))
    G = nx.Graph(adj_matrices[sample, :, :])
    degree_dist = nx.degree(G).values()
    for d in degree_dist:
        degree_vec[d] += 1
    degree_matrices[sample, :] = degree_vec

# save stats as .npy file
np.save(glob.RESULT_PATH+TYPE+'_degree_hist.npy', degree_matrices)
