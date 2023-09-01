"""
Generators for BA homophilic network with minority (a) and majority (b).
inputs $ number_of_nodes, m_links, minority_fraction , h_ab , h_ba


written by: Fariba Karimi
Date: 22-10-2016
"""

import networkx as nx
from collections import defaultdict
import random
import bisect
import copy
import numpy as np


def generate(N, m, f_0, h_00, h_11):
    h_01 = 1 - h_00
    h_10 = 1 - h_11

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.Graph()

    minority = int(f_0 * N)

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
    dist = defaultdict(int)
    H = np.array([[h_00, h_01],
                  [h_10, h_11]])

    #create homophilic distance ### faster to do it outside loop ###
    # for n1 in range(N):
    #     n1_attr = node_attribute[n1]
    #     for n2 in range(N):
    #         n2_attr = node_attribute[n2]
    #         if n1_attr == n2_attr:
    #             if n1_attr == 'minority':
    #                 dist[(n1, n2)] = h_00
    #             else:
    #                 dist[(n1, n2)] = h_11
    #         else:
    #             if n1_attr == 'minority':
    #                 dist[(n1, n2)] = h_01
    #             else:
    #                 dist[(n1, n2)] = h_10
    target_list = list(range(m))
    source = m
    # while source < N:
    #     targets = _pick_targets(G, source, target_list, dist, m)
    #
    #     if targets != set(): #if the node does  find the neighbor
    #         G.add_edges_from(zip([source]*m, targets))
    #
    #     target_list.append(source)
    #     source += 1
    all_edges = []
    while source < N:
        targets = _pick_targets(G, source, target_list, H, m, types)
        # targets = _pick_targets(G, source, target_list, H, dist, m, types)
        if targets != set():  # if the node does  find the neighbor
            all_edges.append(list(zip([source] * m, targets)))
            # G.add_edges_from(zip([source] * m, targets))
        target_list.append(source)
        source += 1
    G.add_edges_from(np.concatenate(all_edges))
    return G

# def _pick_targets(G,source,target_list,dist,m):
#
#     target_prob_dict = {}
#     for target in target_list:
#         target_prob = (dist[(source,target)])* (G.degree(target)+0.000001)
#         target_prob_dict[target] = target_prob


def _pick_targets(G, source, target_list, dist, m, types):
    target_prob_dict = {}
    for target in target_list:
        h = dist[types[source], types[target]]
        # h = dist[(source, target)]
        # print((dist2[(source, target)], h))
        # assert dist2[(source, target)] == h, "OPS"
        target_prob = h * (G.degree(target) + 0.00001)
        target_prob_dict[target] = target_prob

    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = copy.copy(target_list)
    count_looking = 0
    if prob_sum == 0:
        return targets

    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
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





