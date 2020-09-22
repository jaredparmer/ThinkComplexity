""" for chapter 3 of Allen Downey, Think Complexity """

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ch2 import flip, all_pairs


def adjacent_edges(nodes, halfk):
    n = len(nodes)
    for i, u in enumerate(nodes):
        for j in range(i + 1, i + halfk + 1):
            v = nodes[j % n]
            yield u, v


def characteristic_path_length(G):
    """ computes the CPL of graph G, which is the average length of the
    shortest path between each pair of nodes in G """
    return np.mean(list(path_lengths(G)))


def clustering_coefficient(G):
    """ computes clustering coefficient of graph, which is the average of the
    local clustering coefficients of every node """
    cs = [node_clustering(G, node) for node in G]
    return np.nanmean(cs)


def make_ring_lattice(n, k):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(adjacent_edges(nodes, k // 2))
    return G


def make_ws_graph(n, k, p):
    """ returns a Watts-Strogatz graph """
    G = make_ring_lattice(n, k)
    rewire(G, p)
    return G


def node_clustering(G, u):
    """ computes the clustering coefficient for given node u in graph G """
    neighbors = G[u]
    k = len(neighbors)
    if k < 2:
        # local clustering coefficient not defined
        return np.nan

    """ maximum possible number of edges between u and its neighbors, which
    obtains just when all of u's neighbors are connected to each other """
    possible = k * (k - 1) / 2
    exist = 0
    for v, w in all_pairs(neighbors):
        if G.has_edge(v, w):
            exist += 1
    
    return exist / possible


def path_lengths(G):
    """ helper for characteristic_path_length(); generates shortest path
    lengths between every node pair in given graph """
    length_iter = nx.shortest_path_length(G)
    for source, dist_map in length_iter:
        for dest, dist in dist_map.items():
            yield dist


""" order-of-growth analysis in terms of n nodes and m edges: all operations
are constant time, except choice(), which is linear in n. The for loop
executes once for each edge, so m times. O(n*m).
"""
def rewire(G, p):
    """ changes each extant edge in G with probability p, where the new edge
    is chosen with equal chance from all candidates """
    nodes = set(G)
    for u, v in G.edges():
        if flip(p):
            # remove node we're on, and all its extant neighbors
            choices = nodes - {u} - set(G[u])
            new_v = np.random.choice(list(choices))
            G.remove_edge(u, v)
            G.add_edge(u, new_v)


def main():
    # simple ring lattice with n nodes and k adjacent neighbors for each
    n = 10
    k = 4
    lattice = make_ring_lattice(n, k)
    nx.draw_circular(lattice, node_size = 700, with_labels=True)
##    plt.show()

    # Watts-Strogatz (WS) graph with rewire probability p
    p = 0.2
    ws = make_ws_graph(n, k, p)
    nx.draw_circular(ws, node_size = 700, with_labels=True)
##    plt.show()

    # a few WS graphs side by side
    ns = 100
    plt.subplot(1, 3, 1)
    p = 0.0
    ws = make_ws_graph(n, k, p)
    nx.draw_circular(ws, node_size = ns)
    plt.axis('equal')
    print(f"WS Graph({n}, {k}, {p}):")
    print(f"\t C-bar = {clustering_coefficient(ws)}")
    print(f"\t L = {characteristic_path_length(ws)}")

    plt.subplot(1, 3, 2)
    p = 0.2
    ws = make_ws_graph(n, k, p)
    nx.draw_circular(ws, node_size = ns)
    plt.axis('equal')
    print(f"WS Graph({n}, {k}, {p}):")
    print(f"\t C-bar = {clustering_coefficient(ws)}")
    print(f"\t L = {characteristic_path_length(ws)}")

    plt.subplot(1, 3, 3)
    p = 1.0
    ws = make_ws_graph(n, k, p)
    nx.draw_circular(ws, node_size = ns)
    plt.axis('equal')
    print(f"WS Graph({n}, {k}, {p}):")
    print(f"\t C-bar = {clustering_coefficient(ws)}")
    print(f"\t L = {characteristic_path_length(ws)}")

##    plt.show()
    

if __name__ == '__main__':
    main()
