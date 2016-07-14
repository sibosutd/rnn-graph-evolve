"""
Test real-world network data.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import glob
glob.set_dir()

sequences = np.load(glob.DATASET_PATH+'real_seq.npy')

select_index = 5

n_samples, n_timestamps, n_node = sequences.shape
sequence = sequences[select_index]
matrices = np.zeros((n_node, n_node))

D = nx.Graph(np.zeros((n_node, n_node)))
for step in np.arange(n_timestamps):
    timestamp = sequence[step, :]
    indices = timestamp.nonzero()
    if indices[0].shape[0]:
        D.add_edges_from(tuple(indices))
matrices[:, :] = nx.to_numpy_matrix(D)

D = nx.Graph(matrices)
nx.draw(D, node_size=20)
plt.show()
