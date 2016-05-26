#!/usr/bin/env python
"""
Draw a graph with matplotlib, color edges.

You must have matplotlib>=87.7 for this to work.
"""
# Author: Aric Hagberg (hagberg@lanl.gov)

import matplotlib.pyplot as plt
import networkx as nx

G = nx.cycle_graph(20)
pos = nx.spring_layout(G, iterations=200)
# nx.draw(G)
nx.draw(G, pos, edge_color='#A0CBE2', node_color=range(20),
        node_size=800, cmap=plt.cm.Blues, with_labels=True)
# plt.savefig("edge_colormap.png")  # save as png
plt.show()  # display
