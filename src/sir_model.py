import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import otherindex
import SIRsim_HO

'''
程序主要功能
输入：网络图邻接矩阵，需要被设置为感染源的节点序列，感染率，免疫率，迭代次数step
输出：被设置为感染源的节点序列的SIR感染情况---每次的迭代结果（I+R）/n
'''


# def test():
#     B = nx.Graph()
#     # Add nodes with the node attribute "bipartite"
#     # B.add_nodes_from([1, 2, 3, 4], bipartite=0)
#     B.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"
#                          , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"], bipartite=1)
#     # Add edges only between nodes of opposite node sets
#
#     B.add_edges_from([("A", "B"), ("A", "C"), ("B", "C"),
#                       ("B", "T"), ("C", "F"), ("D", "E"),
#                       ("D", "F"), ("E", "F"), ("F", "G"),
#                       ("F", "J"), ("F", "K"), ("G", "J"),
#                       ("G", "K"), ("G", "H"), ("G", "I"),
#                       ("J", "K"), ("K", "L"), ("K", "M"),
#                       ("L", "M"), ("M", "P"), ("M", "N"),
#                       ("M", "O"), ("N", "P"), ("N", "S"),
#                       ("N", "O"), ("O", "P"), ("P", "Q"),
#                       ("P", "R"), ("Q", "R"), ("S", "T"),
#                       ("S", "U"), ("T", "U"), ("T", "V")])
#     # nx.draw_networkx(B, pos=nx.circular_layout(B), with_labels=True, alpha=0.5, node_color='yellow', node_shape='s', )
#     # B = creat_network(read_data())
#     return B
# def generate_graph():
#     graph = nx.Graph()
#     with open('.//Data//structure.csv', 'r') as f:
#         row = csv.reader(f, delimiter=',')
#         next(row)  # 读取首行
#         for r in row:
#             graph.add_edge(r[0], r[1])
#     return graph


def take2(elem):
    return elem[1]


def get_rank_VK(top_n: int):
    with open('..\\'+'Data\\'+'VK'+'.csv', 'r') as f:
        # row = csv.reader(f, delimiter=',')
        lines = f.readlines()
        node_0 = []
        node_05 = []
        node_1 = []
        node_b = []
        node_d = []
        node_c = []
        node_e = []
        node_pr = []
        num = 0  # 行数-1
        for line in lines:
            if (num % 2) == 0:  # num为偶数说明是奇数行
                r = line[:-1].split(',')
                node_0.append([np.double(r[0]), np.double(r[1])])
                node_05.append([np.double(r[0]), np.double(r[2])])
                node_1.append([np.double(r[0]), np.double(r[3])])
                node_b.append([np.double(r[0]), np.double(r[4])])
                node_d.append([np.double(r[0]), np.double(r[5])])
                node_c.append([np.double(r[0]), np.double(r[6])])
                node_e.append([np.double(r[0]), np.double(r[7])])
                node_pr.append([np.double(r[0]), np.double(r[8])])
            else:  # # num为奇数说明是偶数行
                print()
            num += 1
        print(node_0)
        node_0.sort(key=take2, reverse=True)
        node_05.sort(key=take2, reverse=True)
        node_1.sort(key=take2, reverse=True)
        node_b.sort(key=take2, reverse=True)
        node_d.sort(key=take2, reverse=True)
        node_c.sort(key=take2, reverse=True)
        node_e.sort(key=take2, reverse=True)
        node_pr.sort(key=take2, reverse=True)
        top_5 = int(len(node_0) * 0.01 * top_n)
        top_5_node_0 = []
        top_5_node_05 = []
        top_5_node_1 = []
        top_5_node_b = []
        top_5_node_d = []
        top_5_node_c = []
        top_5_node_e = []
        top_5_node_pr = []
        for i in range(top_5):
            top_5_node_0.append(node_0[i][0])
            top_5_node_05.append(node_05[i][0])
            top_5_node_1.append(node_1[i][0])
            top_5_node_b.append(node_b[i][0])
            top_5_node_d.append(node_d[i][0])
            top_5_node_c.append(node_c[i][0])
            top_5_node_e.append(node_e[i][0])
            top_5_node_pr.append(node_pr[i][0])
    return top_5_node_0, top_5_node_05, top_5_node_1, top_5_node_b, top_5_node_d, top_5_node_c, top_5_node_e, top_5_node_pr


def get_rank_USAir(top_n: int):
    with open('..\\'+'Data\\'+'USAir'+'.csv', 'r') as f:
        # row = csv.reader(f, delimiter=',')
        lines = f.readlines()
        node_0 = []
        node_05 = []
        node_1 = []
        node_b = []
        node_d = []
        node_c = []
        node_e = []
        node_pr = []
        num = 0  # 行数-1
        for line in lines:
            if (num % 2) == 0:  # num为偶数说明是奇数行
                r = line[:-1].split(',')
                node_0.append([np.double(r[0]), np.double(r[1])])
                node_05.append([np.double(r[0]), np.double(r[2])])
                node_1.append([np.double(r[0]), np.double(r[3])])
                node_b.append([np.double(r[0]), np.double(r[4])])
                node_d.append([np.double(r[0]), np.double(r[5])])
                node_c.append([np.double(r[0]), np.double(r[6])])
                node_e.append([np.double(r[0]), np.double(r[7])])
                node_pr.append([np.double(r[0]), np.double(r[8])])
            else:  # # num为奇数说明是偶数行
                print()
            num += 1

        # print(node_0)
        node_0.sort(key=take2, reverse=True)
        node_05.sort(key=take2, reverse=True)
        node_1.sort(key=take2, reverse=True)
        node_b.sort(key=take2, reverse=True)
        node_d.sort(key=take2, reverse=True)
        node_c.sort(key=take2, reverse=True)
        node_e.sort(key=take2, reverse=True)
        node_pr.sort(key=take2, reverse=True)
        top_5 = int(len(node_0) * top_n * 0.01)
        top_5_node_0 = []
        top_5_node_05 = []
        top_5_node_1 = []
        top_5_node_b = []
        top_5_node_d = []
        top_5_node_c = []
        top_5_node_e = []
        top_5_node_pr = []
        for i in range(top_5):
            top_5_node_0.append(node_0[i][0])
            top_5_node_05.append(node_05[i][0])
            top_5_node_1.append(node_1[i][0])
            top_5_node_b.append(node_b[i][0])
            top_5_node_d.append(node_d[i][0])
            top_5_node_c.append(node_c[i][0])
            top_5_node_e.append(node_e[i][0])
            top_5_node_pr.append(node_pr[i][0])
    return top_5_node_0, top_5_node_05, top_5_node_1, top_5_node_b, top_5_node_d, top_5_node_c, top_5_node_e, top_5_node_pr


def get_rank_Grid(top_n):
    with open('..\\'+'Data\\'+'Grid'+'.csv', 'r') as f:
        # row = csv.reader(f, delimiter=',')
        lines = f.readlines()
        node_0 = []
        node_05 = []
        node_1 = []
        node_b = []
        node_d = []
        node_c = []
        node_e = []
        node_pr = []
        num = 0  # 行数-1
        for line in lines:
            if (num % 2) == 0:  # num为偶数说明是奇数行
                r = line[:-1].split(',')
                node_0.append([np.double(r[0]), np.double(r[1])])
                node_05.append([np.double(r[0]), np.double(r[2])])
                node_1.append([np.double(r[0]), np.double(r[3])])
                node_b.append([np.double(r[0]), np.double(r[4])])
                node_d.append([np.double(r[0]), np.double(r[5])])
                node_c.append([np.double(r[0]), np.double(r[6])])
                node_e.append([np.double(r[0]), np.double(r[7])])
                node_pr.append([np.double(r[0]), np.double(r[8])])
            else:  # # num为奇数说明是偶数行
                print()
            num += 1
        print(node_0)
        node_0.sort(key=take2, reverse=True)
        node_05.sort(key=take2, reverse=True)
        node_1.sort(key=take2, reverse=True)
        node_b.sort(key=take2, reverse=True)
        node_d.sort(key=take2, reverse=True)
        node_c.sort(key=take2, reverse=True)
        node_e.sort(key=take2, reverse=True)
        node_pr.sort(key=take2, reverse=True)
        top_5 = int(len(node_0) * top_n * 0.01)
        top_5_node_0 = []
        top_5_node_05 = []
        top_5_node_1 = []
        top_5_node_b = []
        top_5_node_d = []
        top_5_node_c = []
        top_5_node_e = []
        top_5_node_pr = []
        for i in range(top_5):
            top_5_node_0.append(node_0[i][0])
            top_5_node_05.append(node_05[i][0])
            top_5_node_1.append(node_1[i][0])
            top_5_node_b.append(node_b[i][0])
            top_5_node_d.append(node_d[i][0])
            top_5_node_c.append(node_c[i][0])
            top_5_node_e.append(node_e[i][0])
            top_5_node_pr.append(node_pr[i][0])
    return top_5_node_0, top_5_node_05, top_5_node_1, top_5_node_b, top_5_node_d, top_5_node_c, top_5_node_e, top_5_node_pr


def get_rank_Lastfm(top_n):
    with open('..\\'+'Data\\'+'LastFM'+'.csv', 'r') as f:
        # row = csv.reader(f, delimiter=',')
        lines = f.readlines()
        node_0 = []
        node_05 = []
        node_1 = []
        node_b = []
        node_d = []
        node_c = []
        node_e = []
        node_pr = []
        num = 0  # 行数-1
        for line in lines:
            if (num % 2) == 0:  # num为偶数说明是奇数行
                r = line[:-1].split(',')
                node_0.append([np.double(r[0]), np.double(r[1])])
                node_05.append([np.double(r[0]), np.double(r[2])])
                node_1.append([np.double(r[0]), np.double(r[3])])
                node_b.append([np.double(r[0]), np.double(r[4])])
                node_d.append([np.double(r[0]), np.double(r[5])])
                node_c.append([np.double(r[0]), np.double(r[6])])
                node_e.append([np.double(r[0]), np.double(r[7])])
                node_pr.append([np.double(r[0]), np.double(r[8])])
            else:  # # num为奇数说明是偶数行
                print()
            num += 1
        # print(node_0)
        node_0.sort(key=take2, reverse=True)
        node_05.sort(key=take2, reverse=True)
        node_1.sort(key=take2, reverse=True)
        node_b.sort(key=take2, reverse=True)
        node_d.sort(key=take2, reverse=True)
        node_c.sort(key=take2, reverse=True)
        node_e.sort(key=take2, reverse=True)
        node_pr.sort(key=take2, reverse=True)
        top_5 = int(len(node_0) * top_n * 0.01)
        top_5_node_0 = []
        top_5_node_05 = []
        top_5_node_1 = []
        top_5_node_b = []
        top_5_node_d = []
        top_5_node_c = []
        top_5_node_e = []
        top_5_node_pr = []
        for i in range(top_5):
            top_5_node_0.append(node_0[i][0])
            top_5_node_05.append(node_05[i][0])
            top_5_node_1.append(node_1[i][0])
            top_5_node_b.append(node_b[i][0])
            top_5_node_d.append(node_d[i][0])
            top_5_node_c.append(node_c[i][0])
            top_5_node_e.append(node_e[i][0])
            top_5_node_pr.append(node_pr[i][0])
    return top_5_node_0, top_5_node_05, top_5_node_1, top_5_node_b, top_5_node_d, top_5_node_c, top_5_node_e, top_5_node_pr


def update_node_status(G, node, beta, gamma):
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
    if G.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            G.nodes[node]['status'] = 'R'
        # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
        if G.nodes[node]['status'] == 'S':
            # 获取当前节点的邻居节点
            # 无向图：G.neighbors(node)
            # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
            neighbors = list(G.neighbors(node))
            # 对当前节点的邻居节点进行遍历
            for neighbor in neighbors:
                # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
                if G.nodes[neighbor]['status'] == 'I':
                    p = random.random()
                    if p < beta:
                        G.nodes[node]['status'] = 'I'
                        break


def count_node(G):
    """
    计算当前图内各个状态节点的数目
    :param G: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in G:
        if G.nodes[node]['status'] == 'S':
            s_num += 1
        elif G.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num


def SIR_network(graph, source, beta, gamma, step):
    """
    获得感染源的节点序列的SIR感染情况
    :param graph: networkx创建的网络
    :param source: 需要被设置为感染源的节点序列
    :param beta: 感染率
    :param gamma: 免疫率
    :param step: 迭代次数
    """
    n = len(graph.nodes)  # 网络节点个数
    sir_values = []  # 存储每一次的感染节点数
    # 初始化节点状态
    for i in range(n):
        graph.nodes[i]['state'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    # 若生成图G中的node编号（从0开始）与节点Id编号（从1开始）不一致，需要减1
    for j in source:
        graph.nodes[j]['state'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(len(source) / n)
    # 开始迭代感染
    for s in range(step):
        # 针对对每个节点进行状态更新以完成本次迭代
        for node in range(n):
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        sir = (i + r) / n  # 本次sir值为迭代结束后 (感染节点数i+免疫节点数r)/总节点数n
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
    return sir_values


if __name__ == '__main__':
    '''
    parameters can be change
    '''
    days = range(1, 16)
    day_num = 15
    num = 1
    repeat_times = 100
    delta = 0.8
    G = otherindex.generate_graph_Grid()
    rank_list = get_rank_Grid(10)

    rate_0 = np.zeros(day_num)
    rate_05 = np.zeros(day_num)
    rate_1 = np.zeros(day_num)
    rate_b = np.zeros(day_num)
    rate_d = np.zeros(day_num)
    rate_c = np.zeros(day_num)
    rate_e = np.zeros(day_num)
    rate_pr = np.zeros(day_num)

    final_0 = []
    final_05 = []
    final_1 = []

    degree_list = list(G.degree)
    avg_degree = 0
    avg_degree_squared = 0

    for node in range(G.number_of_nodes()):
        avg_degree = avg_degree + degree_list[node][1]
        avg_degree_squared = avg_degree_squared + degree_list[node][1] * degree_list[node][1]

    avg_degree = avg_degree / G.number_of_nodes()
    avg_degree_squared = avg_degree_squared / G.number_of_nodes()

    p = num * avg_degree / (avg_degree_squared - avg_degree)
    p_delta = delta * p
    q = 1

    for times in range(repeat_times):
        # 高阶
        # r_0 = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[0])
        # r_05 = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[1])
        # r_1 = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[2])
        # r_b = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[3])
        # r_d = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[4])
        # r_c = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[5])
        # r_e = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[6])
        # r_pr = SIRsim_HO.SIR_sim(G, day_num, p, p_delta, q, rank_list[7])
        # 低阶
        r_0 = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[0])
        r_05 = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[1])
        r_1 = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[2])
        r_b = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[3])
        r_d = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[4])
        r_c = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[5])
        r_e = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[6])
        r_pr = SIRsim_HO.SIR_sim_lo(G, day_num, p, q, rank_list[7])


        for i in range(day_num):
            rate_0[i] += r_0[i]
            rate_05[i] += r_05[i]
            rate_1[i] += r_1[i]
            rate_b[i] += r_b[i]
            rate_d[i] += r_d[i]
            rate_c[i] += r_c[i]
            rate_e[i] += r_e[i]
            rate_pr[i] += r_pr[i]

    for i in range(day_num):
        rate_0[i] = int(rate_0[i] / repeat_times) / G.number_of_nodes()
        rate_05[i] = int(rate_05[i] / repeat_times) / G.number_of_nodes()
        rate_1[i] = int(rate_1[i] / repeat_times) / G.number_of_nodes()
        rate_b[i] = int(rate_b[i] / repeat_times) / G.number_of_nodes()
        rate_d[i] = int(rate_d[i] / repeat_times) / G.number_of_nodes()
        rate_c[i] = int(rate_c[i] / repeat_times) / G.number_of_nodes()
        rate_e[i] = int(rate_e[i] / repeat_times) / G.number_of_nodes()
        rate_pr[i] = int(rate_pr[i] / repeat_times) / G.number_of_nodes()

    plt.title('Grid', fontsize=10)
    plt.xlabel("Time(/day)", fontsize=8)
    plt.ylabel("Rate(%)", fontsize=10)
    plt.tick_params(labelsize=10)
    plt.plot(days, rate_0, label="s=1,low-order", marker='.', markerfacecolor='white', markevery=1, markersize=3,
             alpha=0.5)
    plt.plot(days, rate_05, label="s=0.5", marker='.', markerfacecolor='white', markevery=1, markersize=3)
    plt.plot(days, rate_1, label="s=0,high-order", marker='.', markerfacecolor='white', markevery=1, markersize=3)
    plt.plot(days, rate_b, label="betweeness", marker='.', markerfacecolor='white', markevery=1, markersize=4,
             alpha=0.5)
    plt.plot(days, rate_d, label="degree", marker='.', markerfacecolor='white', markevery=1, markersize=4, alpha=0.5)
    plt.plot(days, rate_c, label="coreness", marker='.', markerfacecolor='white', markevery=1, markersize=4, alpha=0.5)
    plt.plot(days, rate_e, label="eigenector", marker='.', markerfacecolor='white', markevery=1, markersize=4,
             alpha=0.5)
    plt.plot(days, rate_pr, label="pagerank", marker='.', markerfacecolor='white', markevery=1, markersize=4,
             alpha=0.5)
    plt.legend()
    plt.legend(loc=0, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=8)
    # plt.grid()
    # print(rate_0)
    # print(rate_05)
    # print(rate_1)
    # print(rate_b)
    # print(rate_d)
    # print(rate_c)
    # print(rate_e)
    # print(rate_pr)
    #
    plt.show()