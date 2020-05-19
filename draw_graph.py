import numpy as np
import shutil
from glob import glob
import pandas as pd

np.set_printoptions(threshold=np.inf)

def load_npz( npz_path):
  npz_feature = ['node_list', 'graph_node_old', 'graph_node_new', 'graph_edge_old', 'graph_edge_new']

  file = np.load(npz_path)
  old_node = file[npz_feature[1]]  # - old_mean) / (old_std + 1e-6)
  new_node = file[npz_feature[2]]  # - new_mean) / (new_std + 1e-6)
  old_edge = file[npz_feature[3]]
  new_edge = file[npz_feature[4]]

  return old_node, old_edge, new_node, new_edge


if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    import os
    #from graphviz import Digraph
    from networkx.drawing.nx_agraph import graphviz_layout

    m = glob('C:/Users/yinan/Desktop/summary/*/*/*.npz')
    for c in m:
        old_node, old_edge, new_node, new_edge = load_npz(c)
        G1 = nx.Graph()
        x, y = np.nonzero(old_edge)
        egde = np.stack((x,y),axis=1).tolist()
        G1.add_edges_from(egde)
        for i in range(old_edge.shape[0]):
             G1.add_node(i, feature=old_node[i])

        G2 = nx.Graph()
        x, y = np.nonzero(new_edge)
        egde = np.stack((x, y), axis=1).tolist()
        G2.add_edges_from(egde)
        for i in range(new_edge.shape[0]):
            G2.add_node(i, feature=new_node[i])

        pos = graphviz_layout(G1,prog='dot')
        nx.draw(G1, pos)
        h = os.path.split(c)[0] + '/g1.jpg'
        plt.savefig(h)
        pos = graphviz_layout(G2,prog='dot')
        nx.draw(G2, pos)
        h = os.path.split(c)[0] + '/g2.jpg'
        plt.savefig(h)
