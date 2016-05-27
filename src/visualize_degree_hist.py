"""
Visualize degree histogram of all graphs.

Use networkx library.
"""

import numpy as np
import matplotlib.pyplot as plt

import glob

glob.set_dir()
# TYPE = 'BA'
RNN = 'rnn_'  # RNN='' or 'rnn_'

degree_ba = np.load(glob.RESULT_PATH+'BA_'+RNN+'degree_hist.npy')
degree_random = np.load(glob.RESULT_PATH+'RANDOM_'+RNN+'degree_hist.npy')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

# clip the histogram
NUM_BINS = 30
degree_ba = degree_ba[:, :NUM_BINS]
degree_random = degree_random[:, :NUM_BINS]

# the histogram of the data
axes[0].bar(np.arange(NUM_BINS), np.mean(degree_ba, axis=0),
            color='b', yerr=np.std(degree_ba, axis=0))
axes[1].bar(np.arange(NUM_BINS), np.mean(degree_random, axis=0),
            color='r', yerr=np.std(degree_random, axis=0))

axes[0].set_title('BA_'+RNN+'degree distribution')
axes[1].set_title('RANDOM_'+RNN+'degree distribution')
plt.savefig(glob.FIGURE_PATH+RNN+'degree_hist.png')
plt.show()
