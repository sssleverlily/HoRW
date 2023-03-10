from itertools import combinations

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import random
import csv
import pandas as pd
import otherindex as otherindex
import math


# from scipy.sparse import coo_array


def read_data():
    # banchmark 83 nodes 774 edges
    with open("../Data/CrimeNet.txt", "r") as f:
        data = f.readlines()
        edges = []
        edge = []
        for i in data:
            # edges.append([int(i[0: -1].split(" ", 2)[0]), int(i[0: -1].split(" ", 2)[1])])
            edges.append(i[0: -1].split(" ", 2))
        # for j in range(150):
        #     edge.append(edges[j])
    return edges


def creat_network(graph_data: list):
    G = nx.Graph()
    small_graph = graph_data[0:150]  # 先取小的网络来测试
    for item in small_graph:
        G.add_edge(item[0], item[1])
    return G


def calculate_network(G: nx.Graph):
    nx.adjacency_matrix(G)  # 三角形 a-b b-c c-a
    nx.incidence_matrix(G)
    temp = 1
    pairwise_graph = nx.Graph()
    triangle_list = []
    factor_list = []
    five_list = []
    temp_list = []
    # cliques = list(nx.enumerate_all_cliques(G))
    for i in nx.enumerate_all_cliques(G):
        if len(i) == 5:
            five_list.append(i)
            # 删除连边
            for nodei in i:
                pairwise_graph.add_edge(temp, nodei)  # 加边
                for nodej in i:
                    if G.has_edge(nodei, nodej):
                        G.remove_edge(nodei, nodej)
            temp_list.append(temp)
            temp = temp + 1
    # 找到三角形，找到四面体  然后把连边删掉
    for i in nx.enumerate_all_cliques(G):
        if len(i) == 4:
            factor_list.append(i)
            # 删除连边
            for nodei in i:
                pairwise_graph.add_edge(temp, nodei)  # 加边
                for nodej in i:
                    if G.has_edge(nodei, nodej):
                        G.remove_edge(nodei, nodej)
            temp_list.append(temp)
            temp = temp + 1
    for i in nx.enumerate_all_cliques(G):
        if len(i) == 3:
            triangle_list.append(i)
            # 删除连边
            for nodei in i:
                pairwise_graph.add_edge(temp, nodei)  # 加边
                for nodej in i:
                    if G.has_edge(nodei, nodej):
                        G.remove_edge(nodei, nodej)
            temp_list.append(temp)
            temp = temp + 1
    # 再来一次
    for i in nx.enumerate_all_cliques(G):
        if len(i) == 2:
            for node in i:
                pairwise_graph.add_edge(temp, node)  # 加边
            temp_list.append(temp)
            temp = temp + 1
    return pairwise_graph, temp_list, G.nodes


def calculate_network_new(G: nx.Graph):
    nx.adjacency_matrix(G)  # 三角形 a-b b-c c-a
    nx.incidence_matrix(G)
    temp = len(G.nodes) + 1
    pairwise_graph = nx.Graph()
    triangle_list = []
    factor_list = []
    five_list = []
    temp_list = []
    cliques = list(nx.find_cliques(G))
    for c in cliques:
        for nodei in c:
            pairwise_graph.add_edge(temp, nodei)  # 加边
        temp += 1
    temp_list = list(range(len(G.nodes) + 1, temp))
    print(len(G.nodes) + 1, temp + 1)
    return pairwise_graph, temp_list, G.nodes


def toy_example():
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    # B.add_nodes_from([1, 2, 3, 4], bipartite=0)
    B.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"
                         , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"], bipartite=1)
    # Add edges only between nodes of opposite node sets

    B.add_edges_from([("A", "B"), ("A", "C"), ("B", "C"),
                      ("B", "T"), ("C", "F"), ("D", "E"),
                      ("D", "F"), ("E", "F"), ("F", "G"),
                      ("F", "J"), ("F", "K"), ("G", "J"),
                      ("G", "K"), ("G", "H"), ("G", "I"),
                      ("J", "K"), ("K", "L"), ("K", "M"),
                      ("L", "M"), ("M", "P"), ("M", "N"),
                      ("M", "O"), ("N", "P"), ("N", "S"),
                      ("N", "O"), ("O", "P"), ("P", "Q"),
                      ("P", "R"), ("Q", "R"), ("S", "T"),
                      ("S", "U"), ("T", "U"), ("T", "V")])
    # nx.draw_networkx(B, pos=nx.circular_layout(B), with_labels=True, alpha=0.5, node_color='yellow', node_shape='s', )
    # B = creat_network(read_data())
    return B


def change_network(G: nx.Graph, node_list: list, nodes: nx.Graph.nodes):
    hyper_graph = nx.Graph()
    for i in nodes:
        sub_node_list = []
        for j in node_list:
            if G.has_edge(i, j):
                sub_node_list.append(j)
        # 得到了所有连接节点的list,再两两相连就可以了
        for nodei in sub_node_list:
            for nodej in sub_node_list:
                if nodei != nodej:
                    hyper_graph.add_edge(nodei, nodej)
    return hyper_graph


def delete_network(G: nx.Graph):
    for node in list(G.nodes):
        if G.degree(node) <= 900:
            G.remove_node(node)
    return G


# 随机游走的情况
def random_matrix(G: nx.Graph):
    z_matrix = nx.adjacency_matrix(G)
    node_num = z_matrix.shape[0]
    original_matrix = np.zeros((node_num, node_num))
    new_matrix = np.zeros((node_num, node_num))
    degree_list = []
    for i in range(node_num):
        for j in range(node_num):
            original_matrix[i][j] = z_matrix[i, j]
    for i in range(node_num):
        for j in range(node_num):
            if original_matrix[i][j] != 0.0:
                new_matrix[i][j] = 1.0 / (np.sum(original_matrix[i]))
                # (np.sum(original_matrix, axis=1).item(j))
    return new_matrix.transpose(), original_matrix


# 从上到下和从下到上
def hyper_random_matrix(graph: nx.Graph):
    # G是n*m的
    b_graph, temp_list, node = calculate_network_new(graph)
    node_list = []
    m = len(temp_list)
    n = len(node)
    for i in node:
        node_list.append(i)
    a_matrix = np.zeros((m, n))
    u_matrix = np.zeros((m, n))
    v_matrix = np.zeros((n, m))
    # origin_matrix = nx.adjacency_matrix(b_graph)
    print(b_graph.edges)
    # 定义a矩阵
    for i in range(m):
        for j in range(n):
            if b_graph.has_edge(temp_list[i], node_list[j]):
                a_matrix[i][j] = 1
    # a_matrix = np.array(nx.adjacency_matrix(b_graph).todense())
    print(a_matrix)
    # 定义u矩阵
    for i in range(m):
        for j in range(n):
            # if a_matrix[i][j] != 0.0 and np.sum(a_matrix[i]) != 0:
            if a_matrix[i][j] > 0 and np.sum(a_matrix[i]) != 0:
                u_matrix[i][j] = a_matrix[i][j] / np.sum(a_matrix[i])
    # 定义v矩阵
    for i in range(m):
        for j in range(n):
            # if a_matrix[i][j] != 0.0 and np.sum(a_matrix[j]) != 0:
            if a_matrix[i][j] > 0 and np.sum(a_matrix[:, j]) != 0:
                v_matrix[j][i] = a_matrix[i][j] / np.sum(a_matrix[:, j])
    w_matrix = np.dot(v_matrix, u_matrix)
    for j in range(n):
        line_sum = np.sum(w_matrix[:, j])
        for i in range(n):
            w_matrix[i][j] = w_matrix[i][j] / line_sum
    return w_matrix


def hyper_random_matrix_new(graph: nx.Graph):
    b_graph, temp_list, node = calculate_network_new(graph)
    node_list = []
    m = len(temp_list)
    n = len(node)
    for i in node:
        node_list.append(i)
    a_matrix = np.zeros((m, n))
    # 定义a矩阵
    for i in range(m):
        for j in range(n):
            if b_graph.has_edge(temp_list[i], node_list[j]):
                a_matrix[i][j] = 1

    row_sum = np.zeros(m)
    col_sum = np.zeros(n)
    for i in range(m):
        row_sum[i] = 1 / np.sum(a_matrix[i])
    for j in range(n):
        col_sum[j] = 1 / np.sum(a_matrix[:, j])

    u_matrix = a_matrix @ np.diag(col_sum)
    v_matrix = a_matrix.transpose() @ np.diag(row_sum)

    w_matrix = np.dot(v_matrix, u_matrix)

    return w_matrix


def random_walk(origin_matrix: np.ndarray, c_matrix: np.ndarray, t: float):
    rw = c_matrix @ origin_matrix
    while True:
        rw_2 = c_matrix @ rw
        error_matrix = rw_2 - rw
        error = np.linalg.norm(error_matrix, ord=2)
        if error < t:
            return rw
        rw = rw_2
        print(error)


# c = s c_matrix + (1-s) w_matrix
def mix_matrix(c_matrix: np.ndarray, w_matrix: np.ndarray, s: float):
    c = s * c_matrix + (1 - s) * w_matrix
    return c


'''
谱半径<1
'''


def spectral_radius(M):
    a, b = np.linalg.eig(M)  # a为特征值集合，b为特征值向量
    return np.max(np.abs(a))  # 返回谱半径


# rw(t) = c*rw(t-1) = c^t*rw(0)
def mix_walk(origin_matrix: np.ndarray, c_matrix: np.ndarray, t: float):
    rw = c_matrix @ origin_matrix
    while True:
        print(f'rw= {rw}')
        rw_2 = c_matrix @ rw
        print(f'rw_2= {rw_2}')
        error_matrix = np.array(rw_2 - rw, dtype=np.float32)
        error = np.linalg.norm(error_matrix, ord=2)
        # error = spectral_radius(error_matrix)
        print(f'error= {error}')

        if error < t:
            return rw
        rw = rw_2


if __name__ == '__main__':
    graph = otherindex.generate_graph_Grid()
    G = nx.Graph()
    G.add_edges_from(graph.edges())
    c_matrix, origin_matrix = random_matrix(G)  # 随机得到的矩阵
    w_matrix = hyper_random_matrix_new(G)  # 高阶的矩阵
    random_node_tag = []

    print("----------------------")
    test_array = np.ones(len(graph.nodes)).transpose()
    for i in range(len(graph.nodes)):
        test_array[i] = test_array[i] / len(graph.nodes)

    print("00")
    mixx_matrix = mix_matrix(c_matrix, w_matrix, 0)  # 混合的矩阵
    mixx_walk_matrix = mix_walk(test_array, mixx_matrix, 0.01)

    print("05")
    mix3_matrix = mix_matrix(c_matrix, w_matrix, 0.8)  # 混合的矩阵，1相当于全是低阶，0相当于全是高阶
    mix3_walk_matrix = mix_walk(test_array, mix3_matrix, 0.01)

    print("11")
    mix_matrix = mix_matrix(c_matrix, w_matrix, 1)  # 混合的矩阵，1相当于全是低阶，0相当于全是高阶
    mix_walk_matrix = mix_walk(test_array, mix_matrix, 0.01)

    # 全是低阶 1
    mix_node_tag = []
    mix_list = []
    for i in range(mix_walk_matrix.shape[0]):
        mix_list.append(np.sum(mix_walk_matrix[i]))
    mix_dic = {}
    for i in range(mix_walk_matrix.shape[0]):
        mix_dic[list(G.nodes.keys())[i]] = round(mix_list[i] / np.sum(mix_list), 6)
        mix_node_tag.append(round(mix_list[i] / max(mix_list), 6))
    print(mix_dic)
    sum1 = 0
    for key in mix_dic:
        sum1 = sum1 + mix_dic[key]
    print(sum1)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 全是高阶 0
    mixx_node_tag = []
    mixx_list = []
    for i in range(mixx_walk_matrix.shape[0]):
        mixx_list.append(np.sum(mixx_walk_matrix[i]))
    mixx_dic = {}
    for i in range(mixx_walk_matrix.shape[0]):
        mixx_dic[list(G.nodes.keys())[i]] = round(mixx_list[i] / np.sum(mixx_list), 6)
        mixx_node_tag.append(round(mixx_list[i] / max(mixx_list), 6))
    print(mixx_dic)
    sum2 = 0
    for key in mixx_dic:
        sum2 = sum2 + mixx_dic[key]
    print(sum2)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 一半低阶一半高阶 0.5
    mix3_node_tag = []
    mix3_list = []
    for i in range(mix3_walk_matrix.shape[0]):
        mix3_list.append(np.sum(mix3_walk_matrix[i]))
    mix3_dic = {}
    for i in range(mix3_walk_matrix.shape[0]):
        mix3_dic[list(G.nodes.keys())[i]] = round(mix3_list[i] / np.sum(mix3_list), 6)
        mix3_node_tag.append(round(mix3_list[i] / max(mix3_list), 6))
    print(mix3_dic)
    sum3 = 0
    for key in mix3_dic:
        sum3 = sum3 + mix3_dic[key]
    print(sum3)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    betweeness_dic = otherindex.regular_betweeness_centrality(graph)
    degree_dic = otherindex.regular_degree(graph)
    coreness_dic = otherindex.regular_coreness(graph)
    eigenector_dic = otherindex.regular_eigenector_centrality(graph)
    page_rank_dic = otherindex.regular_page_rank(graph)

    f = open('..\\' + 'Data\\' + 'Grid' + '.csv', 'w')
    writer = csv.writer(f)
    print("Write Data----------------------")
    for key, value in mix_dic.items():
        value0 = mixx_dic[key]
        value05 = mix3_dic[key]
        value_b = betweeness_dic[key]
        value_d = degree_dic[key]
        value_c = coreness_dic[key]
        value_e = eigenector_dic[key]
        value_pr = page_rank_dic[key]

        # 顺序 low middle high
        writer.writerow([key, value, value05, value0, value_b, value_d, value_c, value_e, value_pr])
        # writer.writerow([key, value, value05, value0])
    f.close()

'''
画图
'''
# G = test()
# print(G.edges)
# node_color = []
# for i in range(35):
#     node_color.append(random.random())
# B, temp_list, nodes = calculate_network(G)
# hypernet = change_network(B, temp_list, nodes)
# nx.draw(B,
#         pos=nx.bipartite_layout(B, temp_list, align="horizontal"),
#         with_labels=True,
#         alpha=0.7,
#         node_color=node_color,
#         cmap=plt.get_cmap('coolwarm'),
#         node_shape='s')
# plt.show()
# nx.draw_networkx(hypernet)
# plt.show()

'''
'''
