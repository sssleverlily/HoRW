import networkx as nx
import math
import pandas as pd
import numpy as np

'''
度中心性
'''


def regular_degree_centrality(graph: nx.Graph):
    degree_dic = {}
    print("regular_degree_centrality")
    for node in graph.nodes():
        degree_dic[node] = nx.degree(graph) / len(graph.nodes())

    return degree_dic


'''
介数中心性
'''


def regular_betweeness_centrality(graph: nx.Graph):
    betweeness_dic = {}
    betweeness_sum = 0
    # 做一下归一化
    print("regular_betweeness_centrality")
    betweeness_list = nx.betweenness_centrality(graph)
    for i in betweeness_list:
        betweeness_sum = betweeness_sum + betweeness_list[i]
        print(i)
    for i in betweeness_list:
        betweeness_dic[i] = round(betweeness_list[i] / betweeness_sum, 6)
    return betweeness_dic


'''
度
'''


def regular_degree(graph: nx.Graph):
    degree_dic = {}
    degree_sum = 0
    degree_list = nx.degree(graph)
    print("regular_degree")
    for i in list(degree_list._nodes.keys()):
        degree_sum = degree_sum + degree_list[i]
    for i in list(degree_list._nodes.keys()):
        degree_dic[i] = round(degree_list[i] / degree_sum, 6)
    return degree_dic


'''
coreness
'''


def regular_coreness(graph: nx.Graph):
    coreness_dic = {}
    coreness_sum = 0
    # 做一下归一化
    print("regular_coreness")
    coreness_list = nx.core_number(graph)
    for i in list(coreness_list.keys()):
        coreness_sum = coreness_sum + coreness_list[i]
    for i in list(coreness_list.keys()):
        coreness_dic[i] = round(coreness_list[i] / coreness_sum, 6)

    return coreness_dic


'''
特征向量中心性
'''


def regular_eigenector_centrality(graph: nx.Graph):
    eigenector_dic = {}
    eigenector_sum = 0
    # 做一下归一化
    print("regular_eigenector_centrality")
    eigenector_centrality_list = nx.eigenvector_centrality(graph, max_iter=1000)
    for i in list(eigenector_centrality_list.keys()):
        eigenector_sum = eigenector_sum + eigenector_centrality_list[i]
    for i in list(eigenector_centrality_list.keys()):
        eigenector_dic[i] = round(eigenector_centrality_list[i] / eigenector_sum, 12)
    return eigenector_dic


'''
pagerank
'''


def regular_page_rank(graph: nx.Graph):
    return nx.pagerank(graph)


def generate_graph_USAir():
    path = "..\\Data\\USAir.txt"
    graph = nx.Graph()
    lines = pd.read_csv(path)
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))

    return graph


def generate_graph_VK():
    path = "..\\Data\\Valdis_Krebs.txt"
    graph = nx.Graph()
    lines = pd.read_csv(path)
    graph.add_edge(int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1]))
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
    return graph


def generate_graph_Grid():
    path = "..\\Data\\Grid.txt"
    graph = nx.Graph()
    lines = pd.read_csv(path)
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
    return graph


def generate_graph_Lastfm():
    path = "..\\Data\\Lastfm.txt"
    graph = nx.Graph()
    lines = pd.read_csv(path)
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
    return graph
