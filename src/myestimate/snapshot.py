import numpy as np
import torch
import pandas as pd
from utils.lead_lag import get_lead_lag_cluster
import os


class HypergraphSnapshots:
    def __init__(self, symbols, data_dict, train_data_storage, use_cuda):
        self.hypergraph_snapshot = []
        self._use_cuda = use_cuda
        self.data_dict = data_dict
        self._train_data_storage = train_data_storage
        self._symbols = symbols
        self.s = 1.0
        self.upper = True

        ## 1) 섹터기반
        sector_incidence_matrix = self._sector_incidence_matrix()
        sector_hypergraph = self._hypergraph_cora(sector_incidence_matrix)
        self.hypergraph_snapshot.append(sector_hypergraph)

        ## 2) 활동량기반
        activity_matrix = self._activity_matrix()
        activity_hypergraph = self._hypergraph_cora(activity_matrix)
        self.hypergraph_snapshot.append(activity_hypergraph)

        ## 3) 알고리즘기반 DTW
        # dtw_matrix = self._dtw_matrix()
        # dtw_hypergraph = self._hypergraph_cora(dtw_matrix)
        # self.hypergraph_snapshot.append(dtw_hypergraph)

        # 4) 랜덤 구성
        # random_matrix = self._random_matrix()
        # random_hypergraph = self._hypergraph_cora(random_matrix)
        # self.hypergraph_snapshot.append(random_hypergraph)

        # 5) lead lag (pearson correl) 기반
        lead_lag_distance_incidence_matrix = self._lead_lag_distance_incidence_matrix()
        lead_lag_distance_hypergraph = self._hypergraph_cora(lead_lag_distance_incidence_matrix)
        self.hypergraph_snapshot.append(lead_lag_distance_hypergraph)


        #
        # ## 3) lead lag 피어슨 상관계수 기반
        # lead_lag_pearson_incidence_matrix = self._lead_lag_pearson_incidence_matrix()
        # lead_lag_pearson_hypergraph = self._hypergraph_cora(lead_lag_pearson_incidence_matrix)
        # self.hypergraph_snapshot.append(lead_lag_pearson_hypergraph)
        #
        # ## 4) lead lag 켄달 상관계수 기반
        # lead_lag_kendall_incidence_matrix = self._lead_lag_kendall_incidence_matrix()
        # lead_lag_kendall_hypergraph = self._hypergraph_cora(lead_lag_kendall_incidence_matrix)
        # self.hypergraph_snapshot.append(lead_lag_kendall_hypergraph)
        #
        # ## 5) lead lag 상호정보량 기반
        # lead_lag_mutual_information_incidence_matrix = self._lead_lag_mutual_information_incidence_matrix()
        # lead_lag_mutual_information_hypergraph = self._hypergraph_cora(lead_lag_mutual_information_incidence_matrix)
        # self.hypergraph_snapshot.append(lead_lag_mutual_information_hypergraph)

    def _convert_incidence_to_adjacency_torch(self, incidence_matrix):
        num_nodes = incidence_matrix.shape[0]
        num_edges = incidence_matrix.shape[1]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_edges):
                if incidence_matrix[i, j]:
                    connected_nodes_indexes = np.where(incidence_matrix[:, j] == 1)
                    adj_matrix[i, connected_nodes_indexes] = 1
        return torch.Tensor(adj_matrix)


    def _dtw_matrix(self):
        return

    def _activity_matrix(self):
        activity = {
            "group1": ['1201', '1202', '1203', '1204', '3205', '1206'],
            "group2": ['3204', '1205', '3505', '3501', '3502'],
            "group3": ['3207', '3206', '3308', '3208', '3701', '3307', '3209', '3313', '3506', '3304', '3312', '1301'],
            "group4": ['1210', '1207', '1209', '3706', '1208', '3507', '3504', '3707', '1303', '3503', '3704', '1302',
                       '1304'],
            "group5": ['3309', '3703', '1309', '1307', '1306', '3310', '1310', '1305', '1308', '3508', '3708', '3311']
        }
        cat_list = ['group1', 'group2', 'group3', 'group4', 'group5']
        cat_dict = {
            'group1': 0,
            'group2': 1,
            'group3': 2,
            'group4': 3,
            'group5': 4,
        }
        df_list = []
        for group_name, target_list in activity.items():
            for target in target_list:
                df_list.append({'target_id': target, 'group': group_name})

        cat_df = pd.DataFrame(df_list)
        cat_df = cat_df.set_index('target_id')
        incidence_matrix = np.zeros((len(self._symbols), len(cat_list)))

        for i in range(len(self._symbols)):
            cat_key = cat_df.loc[self._symbols[i]].group
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1

        return incidence_matrix

    def _sector_incidence_matrix(self):

        # cat_list = ['group1', 'group2', 'group3']#, 'group4', 'group5', 'group6']
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

        for group_name, target_list in self.data_dict.items():
            for target in target_list:
                df_list.append({'target_id': target, 'group': group_name})

        cat_df = pd.DataFrame(df_list)
        cat_df = cat_df.set_index('target_id')
        incidence_matrix = np.zeros((len(self._symbols), len(cat_list)))

        for i in range(len(self._symbols)):
            cat_key = cat_df.loc[self._symbols[i]].group
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1

        return incidence_matrix

    def _random_matrix(self):
        cat_list = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7', 'group8' ]
        cat_dict = {
            'group1': 0,
            'group2': 1,
            'group3': 2,
            'group4': 3,
            'group5': 4,
            'group6': 5,
            'group7': 6,
            'group8': 7,
        }

        incidence_matrix = np.zeros((len(self._symbols), len(cat_list)))

        for i in range(len(self._symbols)):
            cat_key = np.random.choice(cat_list)
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1

        return incidence_matrix


    def _lead_lag_distance_incidence_matrix(self):
        lead_lag_path = "/home/dyd9800/dataset/exp2_lead_lag.csv"
        if os.path.isfile(lead_lag_path):
            lead_lag = pd.read_csv(lead_lag_path, index_col=0).squeeze("columns")
        else:
            print("no lead-lag")
            # lead_lag = get_lead_lag_cluster(self._train_data_storage)
            # lead_lag.to_csv(lead_lag_path)
        num_cluster = len(lead_lag.unique())
        incidence_matrix = np.zeros((len(self._symbols), num_cluster))
        for i in range(len(self._symbols)):
            cluster_index = lead_lag[int(self._symbols[i])]
            incidence_matrix[i][cluster_index] = 1

        return incidence_matrix

    def _lead_lag_kendall_incidence_matrix(self):
        lead_lag_path = "data/US/sp500/lead_lag_edges_kendall/sp500_leadlag_kendall_{}.csv".format(
            self._start_train_date)
        if os.path.isfile(lead_lag_path):
            lead_lag = pd.read_csv(lead_lag_path, index_col=0).squeeze("columns")
        else:
            lead_lag = get_lead_lag_cluster(self._train_data_storage)
            lead_lag.to_csv(lead_lag_path)
        num_cluster = len(lead_lag.unique())
        incidence_matrix = np.zeros((len(self._symbols), num_cluster))
        for i in range(len(self._symbols)):
            cluster_index = lead_lag[self._symbols[i]]
            incidence_matrix[i][cluster_index] = 1

        return incidence_matrix

    def _lead_lag_mutual_information_incidence_matrix(self):
        lead_lag_path = "data/US/sp500/lead_lag_edges_mutual_information/sp500_leadlag_mutual_information_{}.csv".format(
            self._start_train_date)
        if os.path.isfile(lead_lag_path):
            lead_lag = pd.read_csv(lead_lag_path, index_col=0).squeeze("columns")
        else:
            lead_lag = get_lead_lag_cluster(self._train_data_storage)
            lead_lag.to_csv(lead_lag_path)
        num_cluster = len(lead_lag.unique())
        incidence_matrix = np.zeros((len(self._symbols), num_cluster))
        for i in range(len(self._symbols)):
            cluster_index = lead_lag[self._symbols[i]]
            incidence_matrix[i][cluster_index] = 1

        return incidence_matrix

    def _lead_lag_pearson_incidence_matrix(self):
        lead_lag_path = "data/US/sp500/lead_lag_edges_pearson/sp500_leadlag_pearson_{}.csv".format(
            self._start_train_date)
        if os.path.isfile(lead_lag_path):
            lead_lag = pd.read_csv(lead_lag_path, index_col=0).squeeze("columns")
        else:
            lead_lag = get_lead_lag_cluster(self._train_data_storage)
            lead_lag.to_csv(lead_lag_path)
        num_cluster = len(lead_lag.unique())
        incidence_matrix = np.zeros((len(self._symbols), num_cluster))
        for i in range(len(self._symbols)):
            cluster_index = lead_lag[self._symbols[i]]
            incidence_matrix[i][cluster_index] = 1

        return incidence_matrix

    def _hypergraph_cora(self, incidence_matrix):

        indice_matrix = self._convert_incidence_to_adjacency_torch(incidence_matrix)
        W_e_diag = torch.ones(indice_matrix.size()[1])

        D_e_diag = torch.sum(indice_matrix, 0)
        D_e_diag = D_e_diag.view((D_e_diag.size()[0]))
        D_v_diag = indice_matrix.mm(W_e_diag.view((W_e_diag.size()[0]), 1))
        D_v_diag = D_v_diag.view((D_v_diag.size()[0]))

        Theta = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
                indice_matrix @ torch.diag(W_e_diag) @ \
                torch.diag(torch.pow(D_e_diag, -1)) @ \
                torch.transpose(indice_matrix, 0, 1) @ \
                torch.diag(torch.pow(D_v_diag, -0.5))

        Theta_inverse = torch.pow(Theta, -1)
        Theta_inverse[Theta_inverse == float("Inf")] = 0

        Theta_I = torch.diag(torch.pow(D_v_diag, -0.5)) @ \
                  indice_matrix @ torch.diag(W_e_diag + torch.ones_like(W_e_diag)) @ \
                  torch.diag(torch.pow(D_e_diag, -1)) @ \
                  torch.transpose(indice_matrix, 0, 1) @ \
                  torch.diag(torch.pow(D_v_diag, -0.5))

        Theta_I[Theta_I != Theta_I] = 0
        Theta_I_inverse = torch.pow(Theta_I, -1)
        Theta_I_inverse[Theta_I_inverse == float("Inf")] = 0

        Laplacian = torch.eye(Theta.size()[0]) - Theta

        # fourier_e, fourier_v = torch.symeig(Laplacian, eigenvectors=True) # old version
        fourier_e, fourier_v = torch.linalg.eigh(Laplacian, UPLO='U' if self.upper else 'L')

        wavelets = fourier_v @ torch.diag(torch.exp(-1.0 * fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets_inv = fourier_v @ torch.diag(torch.exp(fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
        wavelets_t = torch.transpose(wavelets, 0, 1)

        wavelets[wavelets < 0.00001] = 0
        wavelets_inv[wavelets_inv < 0.00001] = 0
        wavelets_t[wavelets_t < 0.00001] = 0

        if self._use_cuda:
            wavelets = wavelets.cuda()
            wavelets_inv = wavelets_inv.cuda()

        hypergraph = {"indice_matrix": indice_matrix,
                      "D_v_diag": D_v_diag,
                      "D_e_diag": D_e_diag,
                      "W_e_diag": W_e_diag,
                      "laplacian": Laplacian,
                      "fourier_v": fourier_v,
                      "fourier_e": fourier_e,
                      "wavelets": wavelets,
                      "wavelets_inv": wavelets_inv,
                      "wavelets_t": wavelets_t,
                      "Theta": Theta,
                      "Theta_inv": Theta_inverse,
                      "Theta_I": Theta_I,
                      "Theta_I_inv": Theta_I_inverse,
                      }
        return hypergraph