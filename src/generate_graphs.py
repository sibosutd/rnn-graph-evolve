"""
Graph generation by rnn (load model).

Adapted from char_rnn.py code
"""

import sys
import csv
import igraph
import numpy as np
from keras.models import model_from_json

import glob

glob.set_dir()
TYPE = 'RANDOM'

n_timestamps = 198
temperature = 1.2
NUM_OF_NODE = 100
HIDDEN_UNITS = 256
NUM_LAYER = 2

n_sample = 40

# read model and weights
with open(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)+'_'
          + str(NUM_LAYER)+'_model.json') as f:
    model = model_from_json(f.read())

model.load_weights(glob.MODEL_PATH+TYPE+'_'+str(HIDDEN_UNITS)
                   + '_' + str(NUM_LAYER)+'_weights.h5')

model.compile(optimizer='rmsprop', loss='mse')

# Sample 128 sentences (200 characters each) from model.
# def mnrnd(probs):
#     rnd = np.random.random()
#     for i in xrange(len(probs)):
#         rnd -= probs[i]
#         if rnd <= 0:
#             return i
#     return i

sentences = np.zeros((n_sample, n_timestamps+1, NUM_OF_NODE))
sentences[:, 0, :] = 0

# Start sampling char-sequences. At each iteration i the probability over
# the i-th character of each sequences is computed.
for j in np.arange(n_sample):
    for i in np.arange(n_timestamps):
        if i % 10 == 0:
            print str(j)+' iteration, predict the '+str(i)+' edges'
        probs = model.predict_proba(sentences, verbose=0)[j, i, :]
        probs = probs / np.sum(probs)
        # set temperature to control the tradeoff
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        # Go over each sequence and sample the i-th character.
        select_nodes = np.random.choice(NUM_OF_NODE, 2, replace=False,
                                        p=probs)
        for node in select_nodes:
            sentences[j, i+1, node] = 1
        # sentences[j, i+1, mnrnd(probs[j, :])] = 1
# sentences = [sentence[1:].nonzero()[1] for sentence in sentences]

np.save(glob.DATASET_PATH+TYPE+'_rnn.npy', sentences)

# Convert to readable text.
# print sentences
# np.savetxt(RESULT_PATH+TYPE+'_test.txt', sentences.astype(int))

# sentences_one = sentences[0]

# import csv
# with open(RESULT_PATH+TYPE+'_test.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(sentences)

# graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))  # adj matrix

# edge_count = 0
# sentences_one = sentences_one.astype(int)
# for sentence in sentences_one:
#     # print sentence
#     indices = sentence.nonzero()[0]
#     if len(indices) == 2:
#         graph_matrix[indices[0], indices[1]] = 1
#         graph_matrix[indices[1], indices[0]] = 1
#         edge_count += 1
#     else:
#         print sentence

# g = igraph.Graph.Adjacency(graph_matrix.tolist(), mode=igraph.ADJ_UNDIRECTED)
# print g.degree_distribution()
# print 'num of edges:', g.ecount()

# plot graph figure
# layout = g.layout("kk")
# igraph.plot(g, layout=layout, bbox=(1000, 1000))


# with open(RESULT_PATH+TYPE+'_test.txt', 'w') as f:
#     f.write(sentences)
