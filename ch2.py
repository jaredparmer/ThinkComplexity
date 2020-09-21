""" in-text examples and exercises from ch 2 of Downey, Think Complexity. """

import networkx as nx
import matplotlib.pyplot as plt     # nx uses this to draw figures
import numpy as np


def all_pairs(nodes):
    """ generator function that yields all possible pairs from nodes """
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i > j:
                yield u, v


def flip(p=0.5):
    """ returns True with given probability p, False otherwise """
    # random() from numpy returns a random number between 0 and 1
    return np.random.random() < p


def is_connected(G):
    start = next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)


def make_complete_graph(n):
    """ returns undirected complete Graph """
    """ a complete Graph is one in which every node has an edge with every
    other node. """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G


def make_random_graph(n, p=0.5):
    """ returns undirected Erdos-Renyi Graph with n nodes, and edges between
    each with probability p """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes, p))
    return G


def prob_connected(n, p, iters=100):
    """ estimates the probability that a single ER graph with n nodes, and
    edges between with probability p, is connected """
    tf = [is_connected(make_random_graph(n, p))
          for i in range(iters)]
    return np.mean(tf)


def random_pairs(nodes, p):
    for edge in all_pairs(nodes):
        if flip(p):
            yield edge


def reachable_nodes(G, start):
    """ returns set of nodes that can be seen from start node in graph G """
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen


def foo():
    """ just some garbage to get graphing started. """
    G = nx.DiGraph()
    G.add_node('Alice')
    G.add_node('Bob')
    G.add_node('Chuck')

    G.add_edge('Alice', 'Bob')
    G.add_edge('Alice', 'Chuck')
    G.add_edge('Bob', 'Alice')
    G.add_edge('Bob', 'Chuck')

    ##nx.draw_circular(G,
    ##                 node_size=2000,
    ##                 with_labels=True)
    ##
    ##plt.show()

    positions = dict(Albany=(-74, 43),
                     Boston=(-71, 42),
                     NYC=(-74, 41),
                     Philly=(-75, 40))
    drive_times = {('Albany', 'Boston'): 3,
                   ('Albany', 'NYC'): 4,
                   ('Boston', 'NYC'): 4,
                   ('NYC', 'Philly'): 2}
    G = nx.Graph()
    G.add_nodes_from(positions)
    G.add_edges_from(drive_times)

    nx.draw(G, positions,
            node_shape='s',
            node_size=2500,
            with_labels=True)
    # add edge labels
    nx.draw_networkx_edge_labels(G, positions, edge_labels=drive_times)
    plt.show()

def main():
    n = 10
    ps = np.logspace(-2.5, 0, 11)
    ys = [prob_connected(n, p) for p in ps]
    
##    G = make_random_graph(10, 0.3)
##    nx.draw_circular(G, node_size=1000, with_labels=True)
##    plt.show()

if __name__ == '__main__':
    main()
