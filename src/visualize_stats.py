"""
Visualize statistics of all graphs.

Use networkx library.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

import glob

glob.set_dir()
TYPE = 'RANDOM_rnn'

matrix = np.load(glob.RESULT_PATH+TYPE+'_diam.npy')

n_sample, n_timestamps = matrix.shape

curve = np.zeros((n_timestamps, 3))
curve[:, 1] = np.mean(matrix, axis=0)
curve[:, 0] = np.mean(matrix, axis=0)-np.std(matrix, axis=0)
curve[:, 2] = np.mean(matrix, axis=0)+np.std(matrix, axis=0)

plt.plot(np.arange(n_timestamps), curve[:, 1], 'r', linewidth=2)
plt.fill_between(np.arange(n_timestamps), curve[:, 0], curve[:, 2],
                 facecolor='r', alpha=0.25, linewidth=0)
plt.grid(True)
plt.savefig(glob.RESULT_PATH+TYPE+'_diam.png')
plt.show()
