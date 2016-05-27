"""
Visualize degree distribution of all graphs.

Use networkx library.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

import glob

glob.set_dir()
# TYPE = 'RANDOM'

degree_ba = np.load(glob.RESULT_PATH+'BA_rnn_degree.npy')
degree_random = np.load(glob.RESULT_PATH+'RANDOM_rnn_degree.npy')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))

# rectangular box plot
bplot1 = axes[0].boxplot(degree_ba,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# notch shape box plot
bplot2 = axes[1].boxplot(degree_random,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen']
# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)

# plt.boxplot(matrix)
# plt.grid(True)
plt.savefig(glob.RESULT_PATH+'rnn_degree.png')
plt.show()
