"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np  # for np ops
import torch  # for torch based ops
from scipy.sparse import csr_matrix  # for csr ops

try:
    from os.path import dirname, join

    from cppimport import imp_from_filepath

    # path = join(dirname(__file__), "sources/sampling.cpp")
    path = "code/sources/sampling.cpp"
    sampling = imp_from_filepath(path)
    sampling.seed(2022)
    sample_ext = True
except Exception as e:
    print("Cpp extension not loaded" + str(e))
    sample_ext = False


class MF_DataReader(object):
    """Data Loader class for loading data"""
    def __init__(self,
                 train_set: str,
                 test_set: str,
                 batch_size: int = 64,
                 delimeter: str = ",") -> None:

        # define all the params and initialize the network
        self.batch_size = batch_size
        self.n_user, self.m_item = 0, 0
        self.delimeter = delimeter
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

    @property
    def allPos(self):
        return self.__all_pos

    def all_positive(self, users):
        """Generate all the pos pairs from feedback matrix A"""
        _all_pos = []
        for user in users:
            _all_pos.append(self.train_net[user].nonzero()[-1])
        return _all_pos

    def __read_data(self, filename: str):
        """Function to read and process data into user, item pair"""
        dataSize = 0
        unique_users, users, items = [], [], []
        with open(filename) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(self.delimeter)
                    uid = int(l[0])
                    if self.delimeter == ",":
                        _items = [int(i) for i in l[1].split(" ")]
                    else:
                        _items = [int(i) for i in l[1:]]
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

    def _sample(self, test: bool = False):
        """Function to sample the data into pos and neg pair"""
        if sample_ext:
            S = sampling.sample_negative(
                self.n_user, self.m_item,
                self.train_datasize if not test else self.test_datasizee,
                self.allPos, 1)
            return S
        users = np.random.randint(
            0, self.n_user,
            self.train_datasize if not test else self.test_datasize)
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

    def iterator(self, test: bool = False):
        """Iterator to iterate the data"""
        S = self._sample(test)

        for i in range(S.shape[0] // self.batch_size):
            batch_data = S[i * self.batch_size:(i + 1) * self.batch_size, :]
            yield {
                "user": batch_data[:, 0],
                "items": batch_data[:, 1],
                "labels": batch_data[:, 2]
            }
