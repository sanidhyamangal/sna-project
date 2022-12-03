"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import csv  # for csv writer
import os  # for os related ops
from collections import OrderedDict, defaultdict  # for adding new ops
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd  # for dataframe based loading
from scipy.sparse import csr_matrix


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

    save_data_files(user_item_test,
                    f"{file_dir}/user_item_test.csv",
                    flat_list=True)
    save_data_files(user_item_train,
                    f"{file_dir}/user_item_train.csv",
                    flat_list=True)
    save_data_files(item_suplier_map, f"{file_dir}/item_suplier_map.csv")


def process_dataframe(dataframe: pd.DataFrame, item_suplier_map: dict):
    user_item = defaultdict(list)
    for _, row in dataframe.iterrows():
        user_item[row['buyer']].append(item_suplier_map[(row['item'],
                                                         row['seller'])])

    return user_item


def save_data_files(data_map: dict,
                    filename: str,
                    flat_list: bool = False) -> None:

    with open(filename, "w") as fp:
        for key, val in data_map.items():
            if flat_list:
                val = " ".join(str(i) for i in val)
            fp.write(f"{key},{val}\n")


class MF_DataReader(object):
    def __init__(self, train_set: str, test_set: str) -> None:
        self.n_user, self.m_item = 0, 0
        self.train_users, self.train_items, self.train_unique_users, self.train_datasize = self.__read_data(
            train_set)
        self.test_users, self.test_items, self.test_unique_users, self.test_datasize = self.__read_data(
            test_set)

        self.n_user += 1
        self.m_item += 1

        self.train_net = csr_matrix((np.ones(len(self.train_users)),
                                     (self.train_users, self.train_items)),
                                    shape=(self.n_user, self.m_item))

        self.__all_pos = self.all_positive(list(range(self.n_user)))
        self.allNeg = []
        allItems = set(range(self.m_item))
        for i in range(self.n_user):
            pos = set(self.__all_pos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def allPos(self):
        return self.__all_pos

    def all_positive(self, users):
        _all_pos = []
        for user in users:
            _all_pos.append(self.train_net[user].nonzero()[-1])
        return _all_pos

    def __read_data(self, filename: str):
        dataSize = 0
        unique_users, users, items = [], [], []
        with open(filename) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(",")
                    _items = [int(i) for i in l[1].split()]
                    uid = int(l[0])

                    unique_users.append(uid)
                    users.extend([uid] * len(_items))
                    items.extend(_items)
                    self.m_item = max(self.m_item, max(_items))
                    self.n_user = max(self.n_user, uid)
                    dataSize += len(_items)

        unique_users = np.array(unique_users)
        users = np.array(users)
        items = np.array(items)

        return users, items, unique_users, dataSize

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_items):
            user = self.test_users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def sample(self):
        users = np.random.randint(0, self.n_user, self.train_datasize)
        S = []
        for i, user in enumerate(users):
            y = np.random.randint(0, 2)
            item = self.allPos[user]
            if len(item) == 0:
                continue

            posindex = np.random.randint(0, len(item))
            if y:
                item = item[posindex]
            else:
                while True:
                    negitem = np.random.randint(0, self.m_item)
                    if negitem in item:
                        continue
                    else:
                        break
                item = negitem

            S.append([user, item, y])
        return np.array(S)
