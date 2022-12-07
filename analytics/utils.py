"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for dataframe based loading


def load_dataset(file_path: str):
    # data = np.loadtxt(file_path, sep="::")
    df = pd.read_csv(file_path,
                     sep="::",
                     header=None,
                     names=["buyer", "item", "seller", "timestamp"],engine='python')
    df['timestamp'] = df['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))

    return df


def plot_degree_dist(graph,
                     plot_path: str,
                     plot_name: str,
                     xlim: Tuple[int] = (0, 50)):
    degrees = [graph.degree(n) for n in graph.nodes()]

    plt.hist(degrees, bins="auto")
    plt.xlim(*xlim)
    plt.xlabel("Degrees")
    plt.ylabel("Freq")
    plt.title(plot_name)
    plt.savefig(plot_path)
    plt.clf()
