"""
Visualize degree of all indexed nodes.

Use networkx library.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

import glob

glob.set_dir()
# TYPE = 'RANDOM'
RNN = ''  # RNN='' or 'rnn_'

degree_ba = np.load(glob.RESULT_PATH+'BA_'+RNN+'degree.npy')
degree_random = np.load(glob.RESULT_PATH+'RANDOM_'+RNN+'degree.npy')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

# rectangular box plot
bplot1 = axes[0].boxplot(degree_ba, vert=True)
bplot2 = axes[1].boxplot(degree_random, vert=True)

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.xaxis.set_ticks([])

axes[0].set_title('BA_'+RNN+'average node degree')
axes[1].set_title('RANDOM_'+RNN+'average node degree')

plt.savefig(glob.FIGURE_PATH+RNN+'degree.png')
plt.show()
