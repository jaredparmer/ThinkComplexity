""" in-text examples and exercises from ch 2 of Downey, Think Complexity. """

import networkx as nx
import matplotlib.pyplot as plt     # nx uses this to draw figures
import numpy as np
import random


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


""" exercise 2.2: what's the order of growth for this?

answer: since calling reachable_nodes is O(n + m), while making an iterator and
accessing its first element, getting the length of the set reachable, getting
the length of the Graph, and comparing those lengths is all constant time,
is_connected(G) is also O(n + m).
"""
def is_connected(G):
    start = next(iter(G))
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)


""" for exercise 2.4 """
def m_pairs(nodes, m):
    """ returns m random pairs between given nodes """
    pairs = list(all_pairs(nodes))
    return random.sample(pairs, m)


def make_complete_graph(n):
    """ returns undirected complete Graph """
    """ a complete Graph is one in which every node has an edge with every
    other node. """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return G


""" for exercise 2.4 """
def make_m_graph(n, m):
    """ returns undirected Erdos-Renyi Graph with n nodes, m random edges """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(m_pairs(nodes, m))
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


""" for exercise 2.4 """
def prob_m_connected(n, m, iters=100):
    """ estimates the probability that a single ER graph with n nodes, and
    m random edges, is connected """
    tf = [is_connected(make_m_graph(n, m))
          for i in range(iters)]
    return np.mean(tf)


def random_pairs(nodes, p):
    for edge in all_pairs(nodes):
        if flip(p):
            yield edge


""" cf. section 2.8, the order of growth for this fn with n nodes and m edges:

initializing seen and stack: constant time

within while loop: 
    popping off stack: constant
    checking membership in set: constant
    adding to set: constant
    extending stack: linear in num of neighbors

while loop iterations:
    each node added to seen: n additions in total
    each node added to stack: 2m additions in total

so, O(n + m)
"""
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
    # edge probability at which connectedness probability spikes to 1
    pstar = np.log(n) / n
    # generate probabilities for ER graphs, evenly distributed from
    # 10**(-2.5) to 10**0
    ps = np.logspace(-2.5, 0, 11)
    # estimate prob of being connected for each above probability
    ys = [prob_connected(n, p) for p in ps]

    # print results
##    for p, y in zip(ps, ys):
##        print(p, y)

    # plot results
    plt.axvline(pstar, color='gray')
    plt.plot(ps, ys, color='green')
    plt.show()

    """ for exercise 2.4 """
    m = 15
    # generates number of random edges to include by random proportion of total
    # possible number of edges for n nodes
    ms = [int(p * n * (n-1) / 2) for p in ps]
    ys = [prob_m_connected(n, m) for m in ms]

    # plot results
    plt.axvline(pstar, color='gray')
    plt.plot(ps, ys, color='green')
    plt.show()    
    

if __name__ == '__main__':
    main()
