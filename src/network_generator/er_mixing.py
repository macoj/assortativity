import networkx as nx
import numpy as np


def generate(N, f_0, h_00, h_11, attribute='color', allow_self_loop=False, randomize=True, directed=False):
    n_0 = int(N * f_0)
    n_1 = N - n_0
    types = [0]*n_0 + [1]*n_1
    if randomize:
        np.random.shuffle(types)
    H = np.array([
        [h_00, 1 - h_00],
        [1 - h_11, h_11]])
    self_loop = 1 - int(allow_self_loop)
    edges = []
    for i in range(N):
        for j in range(i + self_loop, N):
            probability = H[types[i], types[j]]
            if probability > np.random.random():
                edges.append((i, j))
    g = nx.Graph()
    g.add_edges_from(edges)
    for n in g.nodes:
        g.nodes[n][attribute] = types[n]
    return g


def generate_n_groups(
        N, b, h, attribute='type', allow_self_loop=False, density_probability=1.0, randomize=True):
    # group sizes
    n = []
    for f_i in b[:-1]:
        n.append(int(N * f_i))
    n.append(N - sum(n))

    # groups assignment
    types = []
    for group in range(len(n)):
        types += [group] * n[group]

    if randomize:
        np.random.shuffle(types)
        
    # probabilities to connect
    intra_group = h
    inter_group = 1 - h

    H = np.diag([intra_group] * len(n))

    H[np.triu_indices(len(n), 1)] = inter_group
    H[np.tril_indices(len(n), -1)] = inter_group
    self_loop = 1 - int(allow_self_loop)
    edges = []
    for i in range(N):
        for j in range(i + self_loop, N):
            if density_probability > np.random.random():
                probability = H[types[i], types[j]]
                if probability > np.random.random():
                    edges.append((i, j))
    g = nx.Graph()
    g.add_edges_from(edges)
    for n in g.nodes:
        g.nodes[n][attribute] = types[n]
    return g




