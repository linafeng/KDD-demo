# -*- coding: utf-8 -*-
import networkx as nx

# 创建有向图
G = nx.DiGraph()
# 有向图之间边的关系,括号的关系是左到右的箭头
edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]
for edge in edges:
    G.add_edge(edge[0], edge[1])
pagerank_list = nx.pagerank(G, alpha=1)

print("pagerank 值是：", pagerank_list)
