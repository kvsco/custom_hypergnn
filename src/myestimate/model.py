import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import HypergraphConv
from utils.layers.temporal_attention import get_subsequent_mask, TemporalAttention
from utils.layers.drnn_models import DLSTM
from utils.layers.hwnn import HWNNLayer
from myestimate.revin import RevIN

class Model(nn.Module):
    def __init__(self, snapshots, num_stock, history_window, lookahead_window, num_feature, embedding_dim=16, rnn_hidden_unit=8,
                 mlp_hidden=16, n_head=4, d_k=8, d_v=8, drop_prob=0.2):
        super(Model, self).__init__()

        self.num_stock = num_stock
        self.lookahead_window = lookahead_window
        self.seq_len = history_window
        self.num_feature = num_feature
        self.rnn_hidden_unit = rnn_hidden_unit
        self.drop_prob = drop_prob
        self.embedding_dim = embedding_dim
        self.mlp_hidden = mlp_hidden
        self.revin = RevIN(1)
        print("using revin normalization.")

        self.snapshots = snapshots
        self.hyper_snapshot_num = len(self.snapshots.hypergraph_snapshot)
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)

        self.embedding = nn.Linear(num_feature, embedding_dim)
        self.lstm1 = DLSTM(embedding_dim * num_stock, self.rnn_hidden_unit * num_stock * 2, num_stock)
        self.ln_1 = nn.LayerNorm(self.rnn_hidden_unit * num_stock * 2)
        self.lstm2 = DLSTM(self.rnn_hidden_unit * num_stock * 2, self.rnn_hidden_unit * num_stock, num_stock)
        self.temp_attn = TemporalAttention(n_head, self.rnn_hidden_unit * num_stock, d_k, d_v, dropout=drop_prob)
        self.dropout = nn.Dropout(self.drop_prob)

        self.hatt1 = HypergraphConv(self.rnn_hidden_unit * self.seq_len, self.rnn_hidden_unit * self.seq_len,
                                    use_attention=False, heads=4, concat=False, negative_slope=0.1, dropout=drop_prob,
                                    bias=True)
        self.hatt2 = HypergraphConv(self.rnn_hidden_unit * self.seq_len, self.rnn_hidden_unit * self.seq_len,
                                    use_attention=False, heads=1, concat=False, negative_slope=0.1, dropout=drop_prob,
                                    bias=True)
        self.convolution_1 = HWNNLayer(self.rnn_hidden_unit * self.seq_len,
                                       self.rnn_hidden_unit * self.seq_len,
                                       self.num_stock,
                                       K1=3,
                                       K2=3,
                                       approx=False,
                                       data=self.snapshots)

        self.convolution_2 = HWNNLayer(self.rnn_hidden_unit * self.seq_len,
                                       self.rnn_hidden_unit * self.seq_len,
                                       self.num_stock,
                                       K1=3,
                                       K2=3,
                                       approx=False,
                                       data=self.snapshots)

        self.mlp_1 = nn.Linear(rnn_hidden_unit * 2, mlp_hidden)
        self.act_1 = nn.ReLU()
        self.mlp_2 = nn.Linear(mlp_hidden, out_features=1)
        # 추가
        self.cnn_decoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, inputs):
        # raw = (batch, node, lookback, feature)
        # target_feature = inputs[...,-3] # tps
        # target_feature = self.revin(target_feature, mode='norm')
        # inputs[...,-3] = target_feature
        inputs = self.embedding(inputs)
        # (batch, node, lookback window, feature_embedding)

        inputs = inputs.permute(0, 2, 1, 3)
        # (batch, lookback window, node, feature_embedding)

        inputs = torch.reshape(inputs, (-1, self.seq_len, self.embedding_dim * self.num_stock))

        slf_attn_mask = get_subsequent_mask(inputs).bool()

        # ---------- 시간 sequence 학습 ----------
        output, _ = self.lstm1(inputs)
        output = self.ln_1(output)
        enc_output, _ = self.lstm2(output)
        enc_output, enc_slf_attn = self.temp_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask.bool())

        enc_output = torch.reshape(enc_output, (-1, self.seq_len, self.num_stock, self.rnn_hidden_unit))
        enc_output = self.dropout(enc_output)
        enc_output = enc_output.permute(0, 2, 1, 3)
        # batch, node, window, encoder_hidden_embedding(16)

        # hyper graph convolution 각 node 관계성 학습
        outputs = []
        for i in range(enc_output.shape[0]):
            x = enc_output[i].reshape(self.num_stock, self.seq_len * self.rnn_hidden_unit)  # 배치 샘플마다loop 배치 키우면 안되겠다.
            channel_feature = []
            for snap_index in range(self.hyper_snapshot_num):
                deep_features_1 = F.leaky_relu(self.convolution_1(x, snap_index, self.snapshots), 0.1)
                deep_features_1 = self.dropout(deep_features_1)
                deep_features_2 = self.convolution_2(deep_features_1, snap_index, self.snapshots)
                deep_features_2 = F.leaky_relu(deep_features_2, 0.1)
                channel_feature.append(deep_features_2)
            deep_features_3 = torch.zeros_like(channel_feature[0])
            for ind in range(self.hyper_snapshot_num):
                deep_features_3 = deep_features_3 + self.par[ind] * channel_feature[ind]
            outputs.append(deep_features_3)

        hyper_output = torch.stack(outputs).reshape(-1, self.num_stock, self.seq_len, self.rnn_hidden_unit)
        # batch, node, window, feature_embedding(16)

        enc_output = torch.cat((enc_output, hyper_output), dim=3)  # 시간 sequence 학습 + node 관계성 학습 concat
        # batch, node, window, feature_embedding(32)

        # ---------- 예측 값 생성 ----------
        # output = self.mlp_1(enc_output)
        # output = F.relu(output) # batch, node, window, feature_embedding(16)
        # output = self.mlp_2(output)
        enc_output = enc_output.permute(0, 1, 3, 2).reshape(-1, 32, self.seq_len)
        output = self.cnn_decoder(enc_output)
        output = output.view(-1, self.num_stock, self.lookahead_window)
        # output = self.revin(output, mode='denorm')

        return output  # shape (batch, node, window)

