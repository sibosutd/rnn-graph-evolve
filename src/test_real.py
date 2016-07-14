"""
Test real-world network data.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import glob
glob.set_dir()

sequences = np.load(glob.DATASET_PATH+'data/real_seq_000.npy')

n_timestamps, n_node = sequences.shape

matrices = np.zeros((n_node, n_node))

D = nx.Graph(np.zeros((n_node, n_node)))
for step in np.arange(n_timestamps):
    timestamp = sequences[step, :]
    indices = timestamp.nonzero()
    D.add_edges_from(tuple(indices))
matrices[:, :] = nx.to_numpy_matrix(D)

D = nx.Graph(matrices)
nx.draw(D, node_size=20)
plt.show()
