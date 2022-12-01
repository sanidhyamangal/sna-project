"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import csv  # for csv writer
import os  # for os related ops
from collections import OrderedDict, defaultdict  # for adding new ops
from datetime import datetime
from typing import Tuple

import pandas as pd  # for dataframe based loading


def create_subfolders_if_not(path: str, dir_struct: bool = False) -> None:
    _path = os.path.split(path)[0]
    if dir_struct:
        _path = path
    os.makedirs(_path, exist_ok=True)


def load_dataset(file_path: str):
    # data = np.loadtxt(file_path, sep="::")
    df = pd.read_csv(file_path,
                     sep="::",
                     header=None,
                     names=["buyer", "item", "seller", "timestamp"])
    df['timestamp'] = df['timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))

    return df


def split_dataset(dataframe: pd.DataFrame,
                  test_size: float = 0.2) -> Tuple[pd.DataFrame]:
    n = dataframe.shape[0]
    split_idx = int((1 - test_size) * n)

    return dataframe.iloc[:split_idx, :], dataframe.iloc[split_idx + 1:, :]


def generate_model_data(dataframe: pd.DataFrame,
                        file_dir: str,
                        test_size: float = 0.2) -> None:
    item_suplier_map = dict()
    create_subfolders_if_not(file_dir, dir_struct=True)

    for idx, row in dataframe.groupby(['item', 'seller'
                                       ]).size().reset_index().iterrows():
        if not item_suplier_map.get((row['item'], row['seller'])):
            item_suplier_map[(row['item'], row['seller'])] = idx

    train_dataframe, test_dataframe = split_dataset(dataframe,
                                                    test_size=test_size)

    user_item_train = process_dataframe(train_dataframe, item_suplier_map)
    user_item_test = process_dataframe(test_dataframe, item_suplier_map)

    save_data_files(user_item_test, f"{file_dir}/user_item_test.csv")
    save_data_files(user_item_train, f"{file_dir}/user_item_train.csv")
    save_data_files(item_suplier_map, f"{file_dir}/item_suplier_map.csv")


def process_dataframe(dataframe: pd.DataFrame, item_suplier_map: dict):
    user_item = defaultdict(list)
    for _, row in dataframe.iterrows():
        user_item[row['buyer']].append(item_suplier_map[(row['item'],
                                                         row['seller'])])

    return user_item


def save_data_files(data_map: dict, filename: str) -> None:

    with open(filename, "w") as fp:
        for key, val in data_map.items():
            fp.write(f"{key},{val}\n")
