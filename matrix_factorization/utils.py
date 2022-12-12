"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import datetime

import pandas as pd
import torch  # for torch based ops


def load_dataset(file_path: str):
    # data = np.loadtxt(file_path, sep="::")
    df = pd.read_csv(file_path,
                     sep="::",
                     header=None,
                     names=["buyer", "item", "seller", "timestamp"])
    df['timestamp'] = df['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))

    return df
