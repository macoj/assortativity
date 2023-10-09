from scipy.optimize import minimize
import networkx as nx


def nx_group_fraction(nx_graph, attribute):
    nodes_group = [nx_graph.nodes[n][attribute] for n in nx_graph.nodes]
    unique, counts = np.unique(nodes_group, return_counts=True)
    return counts, unique


def estimate_h_er_numerical(nx_graph, attribute="color"):
    counts, unique = nx_group_fraction(nx_graph, attribute)
    f_0 = min(counts / counts.sum())
    group_map = dict([(unique[i], i) for i in range(len(unique))])

    absolute_mixing_matrix = nx.attribute_mixing_matrix(nx_graph, attribute, normalized=False, mapping=group_map)
    absolute_mixing_matrix /= 2
    E = absolute_mixing_matrix.sum()
    E_00 = absolute_mixing_matrix[0, 0]
    E_11 = absolute_mixing_matrix[1, 1]
    e_00 = E_00 / E
    e_11 = E_11 / E
    f_1 = 1 - f_0
    f = lambda h_11: np.abs(
        h_11 - (e_11 / e_00) * (f_0 ** 2 / f_1 ** 2) * (f_0 * f_1 + f_1 ** 2 * h_11 + f_0 * f_1 * (1 - h_11)) / (
                    f_0 ** 2 / e_00 - f_0 ** 2 + f_0 * f_1))
    optimization = minimize(f, 0.5, method="L-BFGS-B", bounds=[(0, 1)])
    h_11 = optimization['x'][0]
    h_00 = (e_00 / e_11) * (f_1 ** 2 / f_0 ** 2) * h_11
    return h_00, h_11


def estimate_h_er_analytical(nx_graph, attribute="color"):
    counts, unique = nx_group_fraction(nx_graph, attribute)
    f_0 = min(counts / counts.sum())
    group_map = dict([(unique[i], i) for i in range(len(unique))])

    absolute_mixing_matrix = nx.attribute_mixing_matrix(nx_graph, attribute, normalized=False, mapping=group_map)
    absolute_mixing_matrix /= 2
    E = absolute_mixing_matrix.sum()
    E_00 = absolute_mixing_matrix[0, 0]
    E_11 = absolute_mixing_matrix[1, 1]
    e_00 = E_00 / E
    e_11 = E_11 / E
    f_1 = 1 - f_0

    sum_p_ij = 2 * f_0 * f_1 / (1 - e_00 * (1 - f_1 / f_0) - e_11 * (1 - f_0 / f_1))
    h_11 = e_11 * sum_p_ij / f_1 ** 2
    h_00 = (e_00 / e_11) * (f_1 ** 2 / f_0 ** 2) * h_11
    return h_00, h_11