""" chapter 4 of Downey, Think Complexity """

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import average_clustering
from empiricaldist import Pmf


def degrees(G):
    """ returns list of degrees for each node in given Graph """
    return [G.degree(u) for u in G]


def read_graph(filename):
    G = nx.Graph()
    array = np.loadtxt(filename, dtype=int)
    G.add_edges_from(array)
    return G


def sample_path_lengths(G, nodes=None, trials=1000):
    """ returns the shortest path lengths for random sample of node pairs """
    if nodes is None:
        nodes = list(G)
    else:
        nodes = list(nodes)

    pairs = np.random.choice(nodes, (trials, 2))
    lengths = [nx.shortest_path_length(G, *pair)
               for pair in pairs]

    return lengths


def estimate_path_length(G, nodes=None, trials=1000):
    """ estimates the average shortest path length of a given Graph """
    return np.mean(sample_path_lengths(G, nodes, trials))


fb = read_graph('facebook_combined.txt.gz')
n = len(fb)
m = len(fb.edges())
k_fb = int(round(2*m/n))    # I don't get why we double count the edges, but

print(f"fb Graph nodes = {n}, edges = {m}")
C_fb = average_clustering(fb)
L_fb = estimate_path_length(fb)

""" now construct a WS graph with the same n, k; Downey figured out by trial
and error that when p = 0.05, the C and Ls are comparable. """
ws = nx.watts_strogatz_graph(n, k_fb, 0.05, seed=15)
print(f"Constructing Watts-Strogatz Graph, WS({n}, {k_fb}, 0.05)")
C_ws = average_clustering(ws)
L_ws = estimate_path_length(ws)

print("graph \t n \t C \t L \t mu_k \t sigma_k")
print(f"fb \t {n} \t {C_fb} \t {L_fb} \t "
      f"{np.mean(degrees(fb)):.1f} \t {np.std(degrees(fb)):.1f}")
print(f"WS \t {n} \t {C_ws} \t {L_ws} \t "
      f"{np.mean(degrees(ws)):.1f} \t {np.std(degrees(ws)):.1f}")

""" now use probability mass function objects to check the probability that a
node has a particular degree """
pmf_fb = Pmf.from_seq(degrees(fb))
pmf_ws = Pmf.from_seq(degrees(ws))

# plot the distributions
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
pmf_fb.plot(label='facebook')
plt.xlabel('Degree')
plt.ylabel('PMF')
plt.legend()

plt.subplot(1, 2, 2)
pmf_ws.plot(label='WS graph')
plt.xlabel('Degree')
plt.legend()

plt.savefig('figs/chap04-1')

""" the distribution of probabilities for degree of nodes does not match; so,
we will use the Barabási-Albert (BA) model going forward. This model has the
following parameters:
    n = the number of nodes to generate
    k = the number of edges each node starts with
The BA model adds nodes dynamically, and then connects them to pre-existing
nodes in a preferential way--that is, that is biased in favor of pre-existing
nodes that have many neighbors. """

ba = nx.barabasi_albert_graph(n, int(k_fb/2), seed=15)
pmf_ba = Pmf.from_seq(degrees(ba))

# plot them now in linear scale
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
pmf_fb.plot(label='facebook')
plt.xlabel('Degree')
plt.ylabel('PMF')
plt.legend()

plt.subplot(1, 2, 2)
pmf_ba.plot(label='BA graph')
plt.xlabel('Degree')
plt.legend()

plt.savefig('figs/chap04-2')

# TODO: now in log-log scale
