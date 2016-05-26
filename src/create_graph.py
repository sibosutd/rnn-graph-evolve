"""
Generate graph using different models.

Model: BA and RANDOM
"""

# import igraph
import sys
import numpy as np

NUM_OF_NETWORKS = 5000
NUM_OF_NODE = 100
ER_PROB = 0.04
BA_EDGE = 2
NUM_OF_SEQ = 198    # hardcode for now
TYPE = 'RANDOM'

import glob

glob.set_dir()

sequences = np.zeros((NUM_OF_NETWORKS, NUM_OF_SEQ, NUM_OF_NODE))

for num_iter in np.arange(NUM_OF_NETWORKS):
    print 'The ' + str(num_iter) + ' interations...'
    # node_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))   # record each step
    node_matrix = np.zeros(NUM_OF_NODE)

    graph_matrix = np.zeros((NUM_OF_NODE, NUM_OF_NODE))  # adj matrix

    perm = np.random.permutation(NUM_OF_NODE)  # permute the node order

    # init, start with two connected nodes
    node_vec = np.zeros(NUM_OF_NODE)
    node_vec[perm[0]] = 1
    node_vec[perm[1]] = 1
    node_matrix = np.vstack((node_matrix, node_vec))
    graph_matrix[perm[0], perm[1]] = 1
    graph_matrix[perm[1], perm[0]] = 1
    edge_count = 1

    for index in np.arange(2, NUM_OF_NODE):
        node_index = perm[index]
        if TYPE == 'ER':
            for former_index in np.arange(0, index):
                if np.random.random() < ER_PROB:
                    node_matrix[index, perm[former_index]] += 1
                    node_matrix[index, perm[index]] += 1
                    graph_matrix[perm[index], perm[former_index]] = 1
                    graph_matrix[perm[former_index], perm[index]] = 1
                    edge_count += 1

        elif TYPE == 'BA':
            if index < BA_EDGE:
                for former_index in np.arange(0, index):
                    node_vec = np.zeros(NUM_OF_NODE)
                    node_vec[perm[former_index]] = 1
                    node_vec[perm[index]] = 1
                    node_matrix = np.vstack((node_matrix, node_vec))
                    graph_matrix[perm[index], perm[former_index]] = 1
                    graph_matrix[perm[former_index], perm[index]] = 1
                    edge_count += 1

            else:
                node_prob = np.sum(graph_matrix, axis=0) / \
                    float(np.sum(graph_matrix))
                node_prob = node_prob / np.sum(node_prob)
                select_nodes = np.random.choice(NUM_OF_NODE, BA_EDGE,
                                                replace=False, p=node_prob)
                # note: select_node is not index
                for select_node in select_nodes:
                    node_vec = np.zeros(NUM_OF_NODE)
                    node_vec[select_node] = 1
                    node_vec[perm[index]] = 1
                    node_matrix = np.vstack((node_matrix, node_vec))
                    graph_matrix[perm[index], select_node] = 1
                    graph_matrix[select_node, perm[index]] = 1

                edge_count += BA_EDGE

        elif TYPE == 'RANDOM':
            if index < BA_EDGE:
                for former_index in np.arange(0, index):
                    node_vec = np.zeros(NUM_OF_NODE)
                    node_vec[perm[former_index]] = 1
                    node_vec[perm[index]] = 1
                    node_matrix = np.vstack((node_matrix, node_vec))
                    graph_matrix[perm[index], perm[former_index]] = 1
                    graph_matrix[perm[former_index], perm[index]] = 1
                    edge_count += 1

            else:
                node_prob = np.ones(NUM_OF_NODE)
                node_prob[perm[index]] = 0
                node_prob = node_prob / np.sum(node_prob)
                select_nodes = np.random.choice(NUM_OF_NODE, BA_EDGE,
                                                replace=False, p=node_prob)
                # note: select_node is not index
                for select_node in select_nodes:
                    node_vec = np.zeros(NUM_OF_NODE)
                    node_vec[select_node] = 1
                    node_vec[perm[index]] = 1
                    node_matrix = np.vstack((node_matrix, node_vec))
                    graph_matrix[perm[index], select_node] = 1
                    graph_matrix[select_node, perm[index]] = 1

                edge_count += BA_EDGE

        else:
            print 'no corresponding type.'
            sys.exit(1)

    # print graph_matrix
    # print node_matrix
    # print 'edge_count:', edge_count

    # g = igraph.Graph.Adjacency(graph_matrix.tolist(), mode=igraph.ADJ_UNDIRECTED)
    # print g.degree_distribution()

    # # plot graph figure
    # layout = g.layout("kk")
    # igraph.plot(g, layout=layout, bbox=(1000, 1000))

    sequences[num_iter, :, :] = node_matrix

print sequences.shape

# save sequences
np.save(glob.RESULT_PATH+TYPE+'.npy', sequences)
