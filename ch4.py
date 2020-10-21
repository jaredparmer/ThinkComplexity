""" chapter 4 of Downey, Think Complexity """

import networkx as nx
import numpy as np

def read_graph(filename):
    G = nx.Graph()
    array = np.loadtxt(filename, dtype=int)
    G.add_edges_from(array)
    return G


fb = read_graph('facebook_combined.txt.gz')
n = len(fb)
m = len(fb.edges())


print(n, m)
