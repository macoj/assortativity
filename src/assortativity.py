import numpy as np

def nx_assortativity(nx_graph, attribute):
    import networkx as nx
    return nx.attribute_assortativity_coefficient(nx_graph, attribute)


def nx_minimum_assortativity(nx_graph, attribute):
    import networkx as nx
    counts, unique = nx_group_fraction(nx_graph, attribute)
    group_map = dict([(unique[i], i) for i in range(len(unique))])

    absolute_mixing_matrix = nx.attribute_mixing_matrix(nx_graph, attribute, normalized=False, mapping=group_map)

    e_line = counts * counts.reshape(len(unique), 1) - np.diag(counts)

    absolute_mixing_matrix = np.divide(absolute_mixing_matrix, e_line)
    absolute_mixing_matrix /= absolute_mixing_matrix.sum()

    sum_ai_bi = np.sum(absolute_mixing_matrix.sum(axis=0) * absolute_mixing_matrix.sum(axis=1))

    r_min = - sum_ai_bi / (1 - sum_ai_bi) ## r_min as defined by Newmann

    return r_min


def nx_adjusted_assortativity(nx_graph, attribute):
    import networkx as nx
    counts, unique = nx_group_fraction(nx_graph, attribute)
    group_map = dict([(unique[i], i) for i in range(len(unique))])

    absolute_mixing_matrix = nx.attribute_mixing_matrix(
        nx_graph, attribute, normalized=False, mapping=group_map)

    e_line = counts * counts.reshape(len(unique), 1) - np.diag(counts)

    absolute_mixing_matrix = np.divide(absolute_mixing_matrix, e_line)
    absolute_mixing_matrix /= absolute_mixing_matrix.sum()

    sum_ai_bi = np.sum(absolute_mixing_matrix.sum(axis=0) * absolute_mixing_matrix.sum(axis=1))
    sum_eii = np.sum(absolute_mixing_matrix.diagonal())
    adjusted_assortativity = (sum_eii - sum_ai_bi) / (1 - sum_ai_bi)
    return adjusted_assortativity


def nx_group_fraction(nx_graph, attribute):
    nodes_group = [nx_graph.nodes[n][attribute] for n in nx_graph.nodes]
    unique, counts = np.unique(nodes_group, return_counts=True)
    return counts, unique


def gt_adjusted_assortativity(gt_graph, group_memberships):
    import graph_tool.all as gt
    membership_property = _node_groups(gt_graph, group_memberships)
    unique, counts = np.unique(group_memberships, return_counts=True)
    m, _ = gt.corr_hist(gt_graph, membership_property, membership_property)
    m = np.divide(m, counts.reshape(2, 1)*counts)
    m /= m.sum()
    sum_ai_bi = np.sum(m.sum(axis=0)*m.sum(axis=1))
    sum_eii = np.sum(m.diagonal())
    return (sum_eii - sum_ai_bi)/(1 - sum_ai_bi)


def gt_assortativity(g, group_memberships):
    import graph_tool.all as gt
    return gt.assortativity(g, _node_groups(g, group_memberships))


def _node_groups(g, group_memberships):
    vertex_group = g.new_vertex_property('int')
    for m, v in zip(group_memberships, g.vertices()):
        vertex_group[g.vertex(v)] = m
    return vertex_group


def analytical_assortativity(f_0, h_00, h_11, model="er"):
    assert model in ["er", "ba"], "Model should be either 'er' (Erdos-Renyi) or 'ba' (Barabasi-Albert)"
    if model == "er":
        r = analytical_assortativity_er(f_0, h_00, h_11)
    else:
        r = analytical_assortativity_ba(f_0, h_00, h_11)
    return r


def edge_prob_analytical(h_aa, h_bb, fa):
    #################################
    # it calculates fraction of in-group edges 
    # given the homophily values and 
    # minority size.
    ################################
    
    fb = 1 - fa
    
    h_ab = 1 - h_aa
    h_ba = 1 - h_bb
    
    A = (h_aa - h_ab)*(h_ba - h_bb)
    B = ((2*h_bb - (1-fa) * h_ba)*(h_aa - h_ab) + (2*h_ab - fa*(2*h_aa - h_ab))*(h_ba - h_bb))
    C = (2*h_bb*(2*h_ab - fa*(2*h_aa - h_ab)) - 2*fa*h_ab*(h_ba - h_bb) - 2*(1-fa)*h_ba * h_ab)
    D = - 4*fa*h_ab*h_bb
    
    P = [A, B, C, D]

    for c in np.roots(P):
        if 0<c and c < 2:
            ca = c 
    #ca = np.roots(P)[1] #only one root is acceptable
    #print 'ca',ca
    
    #print 'ca',ca
    
    beta_a = float(fa*h_aa)/ (h_aa*ca + h_ab * (2 - ca)) + float(fb*h_ba)/(h_ba*ca + h_bb*(2-ca))   #beta and ca are accurate
    beta_b = float(fb*h_bb)/ (h_ba*ca + h_bb * (2 - ca)) + float(fa*h_ab)/(h_aa*ca + h_ab*(2-ca)) 

    ba = fa / (1 - beta_a)
    bb = fb / (1 - beta_b)

    
    #print 'beta_a' , beta_a , 'beta_b',beta_b
    
    p_aa = (fa * h_aa *ba)/( h_aa *ba + h_ab *bb)

    p_bb = (fb * h_bb *bb)/( h_bb *bb + h_ba *ba)
    
    p_ab = (fa*h_ab*bb) /(h_ab*bb +h_aa*ba) + (fb*h_ba*ba)/(h_ba*ba +h_bb*bb)
    
    P_aa_analytic = float(p_aa)/(p_aa + p_bb + p_ab) # this is the exact result
    P_bb_analytic = float(p_bb)/(p_aa + p_bb + p_ab)

    #Z = ba
    #K = bb
    
    #P_aa_analytic =(fa * h_aa * Z ) / ((fa * h_aa * Z)+ (fb*(1-h_bb)*Z) + (fa*(1-h_aa)*K)+(fb *h_bb*K) ) #this is a very close approximate
    #P_bb_analytic =(fb * h_bb * K ) / ((fa * h_aa * Z)+ (fb*(1-h_bb)*Z) + (fa*(1-h_aa)*K)+(fb *h_bb*K) ) #this is a very close approximate
    
    return beta_a , beta_b , P_aa_analytic, P_bb_analytic


def analytical_adjusted_assortativity_ba_degree_balance_corrected(fa, h_aa , h_bb):
    beta_a , beta_b, _, _ = edge_prob_analytical(h_aa, h_bb, fa)
   
    fb = 1- fa
    ba = fa / (1 - beta_a) 
    bb = fb / (1 - beta_b) 
    
    h_ab = 1 - h_aa
    h_ba = 1 - h_bb

    p_aa = (fa * h_aa *ba)
    
    p_bb = (fb * h_bb *bb)
    
    p_ab = ( (fa*h_ab*bb)  + (fb*h_ba*ba) ) 
    
    p_aa_norm = p_aa/(fa*ba)
    p_bb_norm = p_bb/(fb*bb)
    
    p_ab_norm =  (fa*h_ab*bb)/(fa*bb)  + (fb*h_ba*ba) /(fb*ba)
    
    E = p_aa_norm + p_bb_norm + p_ab_norm

    A = (p_aa_norm + (1-p_aa_norm))**2 + (p_bb_norm + (1-p_bb_norm))**2 

    B = p_aa_norm + p_bb_norm

    r_norm = ( (1/E) * (B) - (1/E)**2 * (A) ) / ( 1 - (1/E)**2 * (A) )
    
    return r_norm


def analytical_adjusted_assortativity_ba_balance_corrected(fa, h_aa, h_bb):
    beta_a, beta_b, _, _ = edge_prob_analytical(h_aa, h_bb, fa)
    fb = 1 - fa
    ba = fa / (1 - beta_a)
    bb = fb / (1 - beta_b)

    h_ab = 1 - h_aa
    h_ba = 1 - h_bb

    p_aa = (fa * h_aa * ba) / (h_aa * ba + h_ab * bb)

    p_bb = (fb * h_bb * bb) / (h_bb * bb + h_ba * ba)

    p_ab = (fa * h_ab * bb) / (h_ab * bb + h_aa * ba) + (fb * h_ba * ba) / (h_ba * ba + h_bb * bb)

    p_aa_norm = p_aa / (fa * fa)
    p_bb_norm = p_bb / (fb * fb)

    p_ab_norm = p_ab / (fa * fb)  # this is correct

    E = p_aa_norm + p_bb_norm + p_ab_norm
#     A = (p_aa_norm + p_ab_norm / 2) ** 2 + (p_bb_norm + p_ab_norm / 2) ** 2
    A = (p_aa_norm + (1-p_aa_norm)) ** 2 + (p_bb_norm + p_ab_norm / 2) ** 2

        
    B = p_aa_norm + p_bb_norm

    r_norm = ((1 / E) * (B) - (1 / E) ** 2 * (A)) / (1 - (1 / E) ** 2 * (A))

    return r_norm


def analytical_assortativity_ba(f_0, h_00, h_11):
    beta_a, beta_b, _, _ = edge_prob_analytical(h_00, h_11, f_0)

    fb = 1 - f_0
    ba = f_0 / (1 - beta_a)
    bb = fb / (1 - beta_b)

    h_ab = 1 - h_00
    h_ba = 1 - h_11

#     p_aa = f_0 * h_00 * ba
#     p_bb = fb * h_11 * bb
#     p_ab = f_0 * h_ab * bb + fb * h_ba * ba

    p_aa = (f_0 * h_00 *ba)/( h_00 *ba + h_ab *bb) #with denominator the analytical newman is closer to the actual newman
    p_bb = (fb * h_11 *bb)/( h_11 *bb + h_ba *ba)
    p_ab = (f_0*h_ab*bb) /(h_ab*bb +h_00*ba) + (fb*h_ba*ba)/(h_ba*ba +h_11*bb) 

    E = p_aa + p_bb + p_ab
    A = (p_aa + p_ab/2)**2 + (p_bb + p_ab/2)**2
    B = p_aa + p_bb

    r = ((1/E) * B - (1/E)**2 * A) / (1 - (1/E)**2 * A)
    

    
    return r


def analytical_assortativity_er(f_0, h_00, h_11):
    h_01 = 1 - h_00
    h_10 = 1 - h_11
    f_1 = 1 - f_0
    p_00 = h_00 * f_0 ** 2.
    p_01 = h_01 * f_0 * f_1
    p_11 = h_11 * f_1 ** 2.
    p_10 = h_10 * f_1 * f_0
    sum_p_ij = p_00 + p_01 + p_11 + p_10
    p_10 /= sum_p_ij
    p_11 /= sum_p_ij
    p_01 /= sum_p_ij
    p_00 /= sum_p_ij
    sum_a_i_b_i = (p_00 + p_01) * (p_00 + p_10) + (p_11 + p_01) * (p_11 + p_10)
    sum_e_ii = (p_00 + p_11)
    r = (sum_e_ii - sum_a_i_b_i) / (1 - sum_a_i_b_i)
    return r
