"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for dataframe based loading

import networkx as nx
from networkx.algorithms import bipartite


def load_dataset(file_path: str):
    # data = np.loadtxt(file_path, sep="::")
    df = pd.read_csv(file_path,
                     sep="::",
                     header=None,
                     names=["buyer", "item", "seller", "timestamp"], engine='python')
    df['timestamp'] = df['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))

    return df

def create_degree_list(df, col1, col2):
    G_bi = nx.Graph()
    # a and b added to add unique node values
    left_nodes = df[col1].astype(str) + 'a'
    right_nodes = df[col2].astype(str) + 'b'
    G_bi.add_nodes_from(left_nodes, bipartite=0)
    G_bi.add_nodes_from(right_nodes, bipartite=1)
    G_bi.add_edges_from([(left_nodes[idx], right_nodes[idx]) for idx, row in df.iterrows()])
    top_nodes = {n for n, d in G_bi.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(G_bi) - top_nodes
    # print(len(top_nodes))
    # print(len(bottom_nodes))
    deg_X, deg_Y = bipartite.degrees(G_bi, bottom_nodes)
    # print(len(deg_X))

    return list(dict(deg_X).values()), list(dict(deg_Y).values())


def plot_degree_dist(degrees,
                     plot_path: str,
                     plot_name: str,
                     xlim: Tuple[int] = (0, 50)):
    # degrees = [graph.degree(n) for n in graph.nodes()]

    plt.hist(degrees, bins='auto')
    plt.xlim(*xlim)
    plt.xlabel("Degrees")
    plt.ylabel("Freq")
    plt.title(plot_name)
    plt.savefig(plot_path)
    plt.clf()
