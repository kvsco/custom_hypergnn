import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        # self._symbols = ['machine_1', 'machine_2', 'machine_3', 'machine_4', 'machine_5', 'machine_6', 'machine_7', 'machine_8', 'machine_9', 'machine_10', 'machine_11', 'machine_12']

        #['machine_1', 'machine_2', 'machine_3', 'machine_4', 'machine_5', 'machine_6', 'machine_7', 'machine_8', 'machine_9', 'machine_10', 'machine_11', 'machine_12']
        #config['data_dict']["group1"] + config['data_dict']["group2"] + config['data_dict']["group3"] + config['data_dict']["group4"] + config['data_dict']["group5"] + config['data_dict']["group6"]
        self._config = config
        self.lookback_window = config['lookback_window']
        self.lookahead_window = config['lookahead_window']
        self._target_col = config["cols"]
        self._rs_dict = {}
        self._cuda = True

    def get_data_for_machine(self):
        path = '/home/dyd9800/dataset/'
        print("Downloading data...")
        with open(path + "machine_train_storage.pkl", 'rb') as f:
            train_data_storage = pickle.load(f)

        print("PreProcessing data...")
        X_train_storage = []
        y_train_storage = []
        for id in train_data_storage:
            if id not in self._symbols:
                continue

            df = train_data_storage[id][:11600]
            df_raw = df[self._target_col]
            num_train = 8119

            df_data = df_raw.copy()
            train_data = df_data[0:num_train]
            scaler = StandardScaler()
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
            ddf = pd.DataFrame(data, index=df_data.index, columns=df_data.columns)

            train_dataset_indata = self.construct_data(ddf, self._target_col, labels=None)
            # x_train, y_train = self.make_time_dataset(train_dataset_indata) # single step forecast
            x_train, y_train = self.make_time_dataset_for_multi_step(train_dataset_indata) # multi step forecast

            X_train_storage.append(x_train)
            y_train_storage.append(y_train)

        X_full = np.stack((X_train_storage), axis=1)
        y_full = np.stack((y_train_storage), axis=1)
        total_train_dataset = TimeSeriesDataset(X_full, y_full)

        cat_list = ['group1', 'group2', 'group3']
        cat_dict = {
            'group1': 0,
            'group2': 1,
            'group3': 2
        }
        df_list = []

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

        return total_train_dataset, hypergraphsnapshot, incidence_edges[0]


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
            df_raw = df[self._target_col]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

            df_data = df_raw.copy()
            train_data = df_data[0:num_train]
            scaler = StandardScaler()
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
            ddf = pd.DataFrame(data, index=df_data.index, columns=df_data.columns)

            train_dataset_indata = self.construct_data(ddf, self._target_col, labels=None)
            # x_train, y_train = self.make_time_dataset(train_dataset_indata) # single step forecast
            x_train, y_train = self.make_time_dataset_for_multi_step(train_dataset_indata) # multi step forecast

            X_train_storage.append(x_train)
            y_train_storage.append(y_train)

        X_full = np.stack((X_train_storage), axis=1)
        y_full = np.stack((y_train_storage), axis=1)
        total_train_dataset = TimeSeriesDataset(X_full, y_full)


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

        return total_train_dataset, hypergraphsnapshot, incidence_edges[0]

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

        lookahead = self.lookahead_window
        rang = range(slide_win, total_time_len - lookahead, slide_stride)

        for i in rang:
            ft =  data[:, i - slide_win:i]
            tar = data[8, i:i + lookahead]  # multi -step (node_num, slide_win) -3:tps 0:txn_elapse

            x_arr.append(ft)
            y_arr.append(tar)

        x = torch.stack(x_arr).contiguous() # (num_samples, node_num, slide_win)
        y = torch.stack(y_arr).contiguous()
        x = x.transpose(1, 2)               # (num_samples, slide_win, node_num)

        return x, y

    def get_loaders(self, dataset, batch_size):
        dataset_len = int(len(dataset))
        num_train = int(dataset_len * 0.7)
        num_val = int(dataset_len * 0.1)
        num_test = dataset_len - num_train - num_val

        indices = torch.arange(dataset_len)

        # train_indices = indices[:num_train]
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[-1000:]
        combined_indices = torch.cat((train_indices, test_indices))

        train_subset = Subset(dataset, combined_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader
