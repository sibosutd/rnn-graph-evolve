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
TYPE = 'RANDOM'

sequences = np.load(DATASET_PATH+TYPE+'.npy')
sequence = sequences[0]

n_timestamps, NUM_OF_NODE = sequence.shape
print n_timestamps, NUM_OF_NODE

D = nx.Graph(np.zeros((NUM_OF_NODE, NUM_OF_NODE)))

for step in np.arange(1, n_timestamps):
    sample = sequence[step, :]
    indices = sample.nonzero()
    D.add_edges_from(tuple(indices))
    # generate largest connected component
    largest_cc = max(nx.connected_component_subgraphs(D), key=len)
    print nx.diameter(largest_cc)
    nx.draw(largest_cc)
    plt.show()

# nx.draw(D)
# plt.savefig("test.png")
# plt.show()

# print nx.diameter(D)
