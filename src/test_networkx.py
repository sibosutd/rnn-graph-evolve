"""Test networkx."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import glob

glob.set_dir()
TYPE = 'BA'

# a = np.zeros((5, 5))
# D = nx.Graph(a)
# edges = [(0, 1), (2, 1)]
# D.add_edges_from(edges)
# print D.edges()
# # draw
# nx.draw(D)
# # plt.savefig("test.png")
# plt.show()

matrices = np.load(glob.DATASET_PATH+TYPE+'_adj.npy')
print matrices.shape

D = nx.Graph(matrices[0])
nx.draw(D)
plt.show()
