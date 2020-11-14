""" chapter 4 of Downey, Think Complexity """

import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import average_clustering
from empiricaldist import Pmf, Cdf

# for exercise 4.3
import gzip
from ch2 import all_pairs


def _random_subset(repeated_nodes, k):
    """ selects a random subset of nodes without repetition """
    targets = set()
    while len(targets) < k:
        # targets does not yet have enough neighbors in it
        x = random.choice(repeated_nodes)
        # targets is a set, so duplicates are thrown out
        targets.add(x)

    return targets


def barabasi_albert_graph(n, k):
    # generate Graph with k nodes
    G = nx.empty_graph(k)
    # initialize list of neighbors
    targets = list(range(k))
    # initialize list of candidate nodes from which neighbors will be picked
    repeated_nodes = []

    for source in range(k, n):
        # connect node 'source' to all (i.e., k) neighbors in targets list
        G.add_edges_from(zip([source] * k, targets))

        """ update pool so that every node in it is repeated for every neighbor
        it has; e.g., a node with 3 neighbors is there three times """
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * k)

        """ pick k new neighbors for next iteration; the way repeated_nodes is
        updated ensures that a node is picked in proportion to the number of
        neighbors it has """
        targets = _random_subset(repeated_nodes, k)

    return G


def cumulative_prob(pmf, x):
    ps = [pmf[value] for value in pmf if value <= x]
    return np.sum(ps)


def degrees(G):
    """ returns list of degrees for each node in given Graph """
    return [G.degree(u) for u in G]


""" for exercise 4.3 """
def read_actor_network(filename, n=None):
    G = nx.Graph()
    with gzip.open(filename) as f:
        for i, line in enumerate(f):
            nodes = [int(x) for x in line.split()]
            G.add_edges_from(all_pairs(nodes))
            if n and i >= n:
                break

    return G


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
##plt.figure(figsize=(8, 4.5))
##plt.subplot(1, 2, 1)
##pmf_fb.plot(label='facebook')
##plt.xlabel('Degree')
##plt.ylabel('PMF')
##plt.legend()
##
##plt.subplot(1, 2, 2)
##pmf_ws.plot(label='WS graph')
##plt.xlabel('Degree')
##plt.legend()
##
##plt.savefig('figs/chap04-1')
##plt.close()

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

C_ba = average_clustering(ba)
L_ba = estimate_path_length(ba)

print(f"BA \t {n} \t {C_ba} \t {L_ba} \t "
      f"{np.mean(degrees(ba)):.1f} \t {np.std(degrees(ba)):.1f}")

""" now use probability mass function objects to check the probability that a
node has a particular degree """
pmf_fb = Pmf.from_seq(degrees(fb))
pmf_ws = Pmf.from_seq(degrees(ws))

# plot them now in linear scale
##plt.figure(figsize=(8, 4.5))
##plt.subplot(1, 2, 1)
##pmf_fb.plot(label='facebook')
##plt.xlabel('Degree')
##plt.ylabel('PMF')
##plt.legend()
##
##plt.subplot(1, 2, 2)
##pmf_ba.plot(label='BA graph')
##plt.xlabel('Degree')
##plt.legend()
##
##plt.savefig('figs/chap04-2')
##plt.close()

# now in log-log scale
##plt.figure(figsize=(8, 4.5))
##options = dict(ls='', marker='.')
##plt.subplot(1, 2, 1)
##pmf_fb.plot(label='facebook', color='C0', **options)
##plt.xlabel('Degree')
##plt.ylabel('PMF')
##plt.xscale('log')
##plt.yscale('log')
##plt.legend()
##
##plt.subplot(1, 2, 2)
##pmf_ba.plot(label='BA model', color='C2', **options)
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.yscale('log')
##plt.legend()
##
##plt.savefig('figs/chap04-3')
##plt.close()

# using Downey's code to make a BA graph and seeing how it works
##print("Constructing BA(20, 3) graph")
##ba_bespoke = barabasi_albert_graph(20, 3)
##nx.draw_circular(ba_bespoke, node_size=700, with_labels=True)
##plt.show()

""" now use cumulative distribution function objects to represent the data """
cdf_fb = Cdf.from_seq(degrees(fb), name='facebook')
cdf_ws = Cdf.from_seq(degrees(ws), name='WS model')
cdf_ba = Cdf.from_seq(degrees(ba), name='BA model')

# now plot the models on log-x scale to compare with the fb data
##plt.figure(figsize=(8,4.5))
##plt.subplot(1,2,1)
##cdf_fb.plot(color='C0')
##cdf_ws.plot(color='C1')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.ylabel('CDF')
##plt.legend()
##
##plt.subplot(1,2,2)
##cdf_fb.plot(color='C0')
##cdf_ba.plot(color='C2')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.legend()
##
##plt.savefig('figs/chap04-4')
##plt.close()

""" this shows that the WS model is very bad, and the BA is okay from the
median and up. Now we'll use the Complementary CDF to get a closer look at the
BA model's performance.

Note: if the underlying PMF obeys the power law, then the CCDF will, too. This
implies further that the CCDF will be a straight line on log-log scale. """
##plt.figure(figsize=(8,4.5))
##plt.subplot(1,2,1)
##(1 - cdf_fb).plot(color='C0')
##(1 - cdf_ws).plot(color='C1')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.ylabel('CCDF')
##plt.yscale('log')
##plt.legend()
##
##plt.subplot(1,2,2)
##(1 - cdf_fb).plot(color='C0')
##(1 - cdf_ba).plot(color='C2')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.yscale('log')
##plt.legend()
##
##plt.savefig('figs/chap04-5')
##plt.close()

""" exercise 4.2. We'll now try to use the Holme-Kim algorithm to generate
a growing graph that approximates the fb data's features. Such a graph obeys
the power law. """

""" the key difference from the BA model is that there is a given probability
that a new node, after getting a random edge and thus a neighbor, also gets an
edge to one of its neighbor's neighbors, thus forming a 'triangle'. """

# using p=1.0 to maximize clustering, still not high enough to match fb
hk = nx.powerlaw_cluster_graph(n, int(k_fb/2), p=1, seed=15)

C_hk = average_clustering(hk)
L_hk = estimate_path_length(hk)

print(f"HK \t {n} \t {C_hk} \t {L_hk} \t "
      f"{np.mean(degrees(hk)):.1f} \t {np.std(degrees(hk)):.1f}")

# now compare distributions
cdf_hk = Cdf.from_seq(degrees(hk), name='HK model')

##plt.figure(figsize=(10,4.5))
##plt.subplot(1,2,1)
##cdf_fb.plot(color='C0')
##cdf_hk.plot(color='C1')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.ylabel('CDF')
##plt.legend()
##
##plt.subplot(1,2,2)
##(1 - cdf_fb).plot(color='C0')
##(1 - cdf_hk).plot(color='C1')
##plt.xlabel('Degree')
##plt.xscale('log')
##plt.yscale('log')
##plt.ylabel('CCDF')
##plt.legend()
##
##plt.savefig('figs/chap04-6')
##plt.close()

""" upshot: the HK model's distribution is about as good as the BA model's, but
it has higher clustering. """

""" Exercise 4.3. Similar sort of analysis on actor network data, which is what
Barabási and Albert used in their original presentation of the model. """

actors = read_actor_network('actor.dat.gz', n=10000)
C_actors = average_clustering(actors, trials=10000)

print()
print("Actor collaboration data.")
print(f"actors \t {n} \t {C_actors} \t "
      f"{np.mean(degrees(actors)):.1f} \t {np.std(degrees(actors)):.1f}")

# not plot it
pmf_actors = Pmf.from_seq(degrees(actors), name='actors')
cdf_actors = Cdf.from_seq(degrees(actors), name='actors')

# plot them now in log scale
plt.figure(figsize=(12, 4.5))
plt.subplot(1, 3, 1)
options = dict(ls='', marker='.')
pmf_actors.plot(label='actors', **options)
plt.xlabel('Degree')
plt.ylabel('PMF')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.subplot(1, 3, 2)
cdf_actors.plot(label='actors')
plt.xlabel('Degree')
plt.xscale('log')
plt.ylabel('CDF')
plt.legend()

plt.subplot(1, 3, 3)
(1 - cdf_actors).plot(label='actors')
plt.xlabel('Degree')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('CCDF')
plt.legend()

plt.savefig('figs/chap04-7')
plt.close()
