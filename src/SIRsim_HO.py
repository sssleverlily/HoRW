import copy
import random
import networkx as nx
import numpy as np
import itertools
import otherindex
from collections import Counter
from itertools import chain
from collections import defaultdict


# 找到网络中的所有三角形list
def triangle_list(G):
    triangle_list = []
    # triangle_list = list(frozenset(c) for c in nx.find_cliques(G) if len(c) == 3)
    for i in nx.enumerate_all_cliques(G):
        if len(i) == 3:
            triangle_list.append(i)
        if len(i) > 3:
            break
    return triangle_list


def updateNetworkState_HO(G, beta, beta_delta, gamma, cumulative_infected, now_infected, recovered, triangle_dic):
    copy_infected = set(now_infected)
    new_infected = set()
    remove_infected = set()
    be_simulate = set(G.nodes) - recovered - copy_infected
    for node in be_simulate:  # 遍历图中处于S态的节点，每一个节点状态进行更新
        # 先算三角形
        triangle_order = 0
        if len(triangle_dic[node]) != 0:
            for triangle in triangle_dic[node]:
                is_activate = 1  # 默认是1，如果另外2个节点中的任意一个不是I状态，那么就是0
                for node_in_it in triangle:
                    if node_in_it != node and node_in_it not in now_infected:  # [1/0, 1, 1] 才active
                        is_activate = 0
                triangle_order += is_activate  # 最终有几个三角形就是几阶

        # 再找边
        pair_order = 0
        for neighbor_node in set(G.neighbors(node)):
            if neighbor_node in now_infected:
                pair_order += 1

        beta_sum = 1 - ((1 - beta) ** pair_order) * ((1 - beta_delta) ** triangle_order)

        if random.random() < beta_sum:
            new_infected.add(node)

    for node in copy_infected:
        if random.random() < gamma:
            remove_infected.add(node)
            now_infected.remove(node)

    # now_infected = now_infected - remove_infected
    now_infected.extend(new_infected)
    recovered |= remove_infected  # 两个set不重复拼接
    cumulative_infected.extend(new_infected)


def updateNetworkState_LO(G, beta, gamma, cumulative_infected, now_infected, recovered):
    copy_infected = set(now_infected)
    new_infected = set()
    remove_infected = set()
    be_simulate = set(G.nodes) - recovered - copy_infected
    for node in be_simulate:  # 遍历图中处于S态的节点，每一个节点状态进行更新
        # 再找边
        pair_order = 0
        for neighbor_node in set(G.neighbors(node)):
            if neighbor_node in now_infected:
                pair_order += 1

        beta_sum = 1 - ((1 - beta) ** pair_order)

        if random.random() < beta_sum:
            new_infected.add(node)

    for node in copy_infected:
        if random.random() < gamma:
            remove_infected.add(node)
            now_infected.remove(node)

    # now_infected = now_infected - remove_infected
    now_infected.extend(new_infected)
    recovered |= remove_infected  # 两个set不重复拼接
    cumulative_infected.extend(new_infected)


def parameter_vc(G, gamma, mu, triangle_list):
    degree_sum = 0
    simplex_sum = 0
    triangle_list_one = list(chain.from_iterable(triangle_list))
    for item in list(nx.degree(G)):
        degree_sum = degree_sum + item[1]

    result = Counter(triangle_list_one)
    k = degree_sum / nx.number_of_nodes(G)
    k_delta = sum(result) / nx.number_of_nodes(G)
    return (gamma + mu) * k / (gamma * k_delta)


'''
beta:感染率
gamma:恢复率

I 是感染的节点数
Recovered 是恢复的节点数

G, beta, gamma, mu, cumulative_infected, now_infected, recovered, recover_to_sus,
                       higher_order_adjacency_matrix, triangle_list, p_s_init, p_i_init, p_r_init
'''


def SIR_sim(G, days, beta, beta_delta, gamma, initial_infected):
    now_I = copy.deepcopy(initial_infected)
    cumulative_I = copy.deepcopy(initial_infected)
    I = []
    Recovered = []
    R = set()
    tri_neighbors_dict = get_tri_neighbors_dict(triangle_list(G))
    for t in range(0, days):
        updateNetworkState_HO(G, beta, beta_delta, gamma, cumulative_I, now_I, R, tri_neighbors_dict)  # 对网络状态进行模拟更新
        I.append(len(cumulative_I))
        Recovered.append(len(R))
    return I


# 非高阶传播
def SIR_sim_lo(G, days, beta, gamma, initial_infected):
    now_I = copy.deepcopy(initial_infected)
    cumulative_I = copy.deepcopy(initial_infected)
    I = []
    Recovered = []
    R = set()
    for t in range(0, days):
        updateNetworkState_LO(G, beta, gamma, cumulative_I, now_I, R)  # 对网络状态进行模拟更新
        I.append(len(cumulative_I))
        Recovered.append(len(R))
    return Recovered


def get_tri_neighbors_dict(triangles_list):
    tri_neighbors_dict = defaultdict(list)
    for i, j, k in triangles_list:
        tri_neighbors_dict[i].append((j, k))
        tri_neighbors_dict[j].append((i, k))
        tri_neighbors_dict[k].append((i, j))
    return tri_neighbors_dict
