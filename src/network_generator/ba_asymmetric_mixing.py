import networkx as nx
from collections import defaultdict
import random
import bisect
import copy
import numpy as np


def generate(N, m, minority_fraction, h_00, h_11):
    G = nx.Graph()
    minority = int(minority_fraction * N)

    minority_nodes = random.sample(range(N), minority)
    node_attribute = {}
    types = []
    for n in range(N):
        if n in minority_nodes:
            G.add_node(n, color='red')
            node_attribute[n] = 'minority'
            types.append(0)
        else:
            G.add_node(n, color='blue')
            node_attribute[n] = 'majority'
            types.append(1)

    H = np.array([[1 - h_00, h_00], 
                  [h_11, 1 - h_11]])
    target_list = list(range(m))
    source = m  # start with m nodes

    while source < N:

        targets = _pick_targets(G, source, target_list, H, m, types)

        if targets != set():  # if the node does  find the neighbor
            G.add_edges_from(zip([source] * m, targets))

        target_list.append(source)
        source += 1

    return G

def _pick_targets(G, source, target_list, dist, m, types):
    target_prob_dict = {}
    for target in target_list:
        h = dist[types[source], types[target]]
        target_prob = (1 - h) * (G.degree(target) + 0.00001)
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets  # it returns an empty set

    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G):  # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets
