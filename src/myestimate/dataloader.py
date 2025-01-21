import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

from myestimate.snapshot import HypergraphSnapshots
from scipy import sparse
from torch_geometric import utils
import pandas as pd
import os
import random
import torch

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_item = torch.FloatTensor(self.X[idx])
        y_item = torch.FloatTensor(self.y[idx])
        return X_item, y_item


class MyDataLoader():
    def __init__(self, config):
        self._symbols = config['data_dict']["group1"] + config['data_dict']["group2"] + config['data_dict']["group3"] + config['data_dict']["group4"] + config['data_dict']["group5"] + config['data_dict']["group6"]
        self._config = config
        self.lookback_window = 20
        self._target_col = config["cols"]
        self._rs_dict = {}
        self._cuda = True

    def get_data(self):
        path = '/home/dyd9800/dataset/'
        suffix = '.parquet'  # file format
        train_data_storage = {}
        scaler = joblib.load('/home/dyd9800/artifact/scaler.pkl')

        # 1. data loading
        print("Downloading data...")
        train_data_storage_1 = {}
        train_data_storage_2 = {}
        train_data_storage_3 = {}
        test_data_storage_1 = {}
        test_data_storage_2 = {}
        test_data_storage_3 = {}

        with open(path + "train_storage_1.pkl", 'rb') as f:
            train_data_storage_1 = pickle.load(f)

        with open(path + "train_storage_2.pkl", 'rb') as f:
            train_data_storage_2 = pickle.load(f)

        with open(path + "train_storage_3.pkl", 'rb') as f:
            train_data_storage_3 = pickle.load(f)

        with open(path + "test_storage_1.pkl", 'rb') as f:
            test_data_storage_1 = pickle.load(f)

        with open(path + "test_storage_2.pkl", 'rb') as f:
            test_data_storage_2 = pickle.load(f)

        with open(path + "test_storage_3.pkl", 'rb') as f:
            test_data_storage_3 = pickle.load(f)

        train_data_storage = {**train_data_storage_1, **train_data_storage_2, **train_data_storage_3}
        test_data_storage = {**test_data_storage_1, **test_data_storage_2, **test_data_storage_3}

        # 2. data_preprocessing
        print("PreProcessing data...")
        X_train_storage = []
        y_train_storage = []
        for id in train_data_storage:
            if id not in self._symbols:
                continue

            df = train_data_storage[id]
            df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

            train_dataset_indata = self.construct_data(df_scaled, self._target_col, labels=None)
            x_train, y_train = self.make_time_dataset(train_dataset_indata)

            X_train_storage.append(x_train)
            y_train_storage.append(y_train)

        X_full = np.stack((X_train_storage), axis=1)
        y_full = np.stack((y_train_storage), axis=1)
        total_train_dataset = TimeSeriesDataset(X_full, y_full)

        X_train_storage = []
        y_train_storage = []
        for id in test_data_storage:
            if id not in self._symbols:
                continue
            df = test_data_storage[id]
            df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

            test_dataset_indata = self.construct_data(df_scaled, self._target_col, labels=None)
            x_train, y_train = self.make_time_dataset(test_dataset_indata)

            X_train_storage.append(x_train)
            y_train_storage.append(y_train)

        X_full = np.stack((X_train_storage), axis=1)
        y_full = np.stack((y_train_storage), axis=1)
        total_test_dataset = TimeSeriesDataset(X_full, y_full)

        cat_list = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6']
        cat_dict = {
            'group1': 0,
            'group2': 1,
            'group3': 2,
            'group4': 3,
            'group5': 4,
            'group6': 5,
        }
        df_list = []

        # 훈련 데이터와 incidence_matrix 순서 동일하게 (*중요)
        for group_name, target_list in self._config["data_dict"].items():
            for target in target_list:
                df_list.append({'target_id': target, 'group': group_name})

        cat_df = pd.DataFrame(df_list)
        cat_df = cat_df.set_index('target_id')
        incidence_matrix = np.zeros((len(self._symbols), len(cat_list)))

        for i in range(len(self._symbols)):
            cat_key = cat_df.loc[self._symbols[i]].group
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1

        inci_sparse = sparse.coo_matrix(incidence_matrix)
        incidence_edges = utils.from_scipy_sparse_matrix(inci_sparse)
        hypergraphsnapshot = HypergraphSnapshots(self._symbols, self._config["data_dict"], train_data_storage, self._cuda)
        print("Done!")

        return total_train_dataset, total_test_dataset, hypergraphsnapshot, incidence_edges[0]

    def construct_data(self, data, feature_map, labels=None):
        res = []

        for feature in feature_map:
            if feature in data.columns:
                res.append(data.loc[:, feature].values.tolist())
            else:
                print(feature, 'not exist in data')
        # no label
        return res

    def make_time_dataset(self, raw_data):
        x_data = raw_data[:]
        data = torch.tensor(x_data).float()

        x_arr, y_arr = [], []
        slide_win, slide_stride = self.lookback_window, 1
        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride)

        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[:, i]                   # single-step (node_num,)

            x_arr.append(ft)
            y_arr.append(tar)

        x = torch.stack(x_arr).contiguous() # (num_samples, node_num, slide_win)
        y = torch.stack(y_arr).contiguous()
        x = x.transpose(1, 2)               # (num_samples, slide_win, node_num)

        return x, y

    def make_time_dataset_for_multi_step(self, raw_data):
        x_arr, y_arr = [], []
        slide_win, slide_stride = self.lookback_window, 1

        x_data = raw_data[:]
        data = torch.tensor(x_data).float()
        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len-slide_win + 1, slide_stride)

        for i in rang:
            ft =  data[:, i-slide_win : i]
            tar = data[:, i : i+slide_win]  # multi -step (node_num, slide_win)

            x_arr.append(ft)
            y_arr.append(tar)

        x = torch.stack(x_arr).contiguous() # (num_samples, node_num, slide_win)
        y = torch.stack(y_arr).contiguous()
        x = x.transpose(1, 2)               # (num_samples, slide_win, node_num)
        y = y.transpose(1, 2)

        return x, y

    def get_loaders(self, train_dataset, test_dataset, batch_size, val_ratio=0.2):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
