"""Test networkx."""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

a = np.reshape(np.random.random_integers(0, 1, size=25), (5, 5))
D = nx.Graph(a)

print len(D.edges())

nx.draw(D)
plt.savefig("test.png")
plt.show()
