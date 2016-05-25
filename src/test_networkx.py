"""Test networkx."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

a = np.zeros((5, 5))
D = nx.Graph(a)
edges = [(0, 1), (2, 1)]
D.add_edges_from(edges)
print D.edges()
# draw
nx.draw(D)
# plt.savefig("test.png")
plt.show()
