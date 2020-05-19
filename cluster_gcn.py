import numpy as np
import torch
from partition_utils import partition_graph
import partition_utils
import scipy.sparse as sp
import random

def create_partition(edge, node_feature, p=50, q=1):
	# Pooling
	edge_pool = torch.zeros(edge[0].shape)
	for adj in edge:
	    edge_pool[adj.nonzero()] = 1
	# Partitioning
	idx_nodes = np.array([n for n in range(edge_pool.shape[0])], dtype=np.int32)
	part_adj, parts = partition_graph(sp.csr_matrix(edge_pool), idx_nodes,p)

	# Creating subgraph from randomly chosen clusters
	batch = []
	random.shuffle(parts)
	for idx in range(q):
	    while len(parts[idx]) == 0: # Ignore empty clusters
	        idx += 1
	    for node in parts[idx]:
	        batch.append(node)
	new_edge = torch.zeros((edge.shape[0],len(batch),len(batch)))
	new_node_feature = torch.zeros((node_feature.shape[0],len(batch),node_feature.shape[2]))
	for i in range(edge.shape[0]):
	    new_edge[i] = edge[i][batch][:,batch]
	    new_node_feature[i] = node_feature[i][batch]
	edge = new_edge
	node_feature = new_node_feature
	return edge, node_feature