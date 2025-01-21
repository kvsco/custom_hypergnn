import os
import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
from datetime import datetime


from myestimate.dataloader import MyDataLoader
from myestimate.model import Model


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
        self.threshold = 0.05   # minmax scaler 0~1
        self.weight = nn.Parameter(torch.tensor(10.0))  # learnable params

    def forward(self, preds, targets):
        small_value_mask = (targets < self.threshold).float()
        # MSE 계산
        small_value_loss = small_value_mask * self.weight * (preds - targets) ** 2
        large_value_loss = (1 - small_value_mask) * (preds - targets) ** 2
        loss = small_value_loss + large_value_loss
        return loss.mean()


class Trainer():
    def __init__(self, config):
        self._config = config

        # group cluster num
        self._symbols = config['data_dict']["group1"] + config['data_dict']["group2"] + config['data_dict']["group3"] + config['data_dict']["group4"] + config['data_dict']["group5"] + config['data_dict']["group6"]

        # model parameters
        self._hidden_dim = 128
        self._dropout = 0.3
        self._batch_size = 32
        self._epochs = 200
        self._cuda = True  # in gpu server
        self._earlystop = 10
        self._eval_iter = 10
        self._rnn_units = 16

        # Get data
        self.lookback_window = 20
        self._data_loader = MyDataLoader(self._config)
        self.train_dataset, self.test_dataset, self._hypergraphsnapshot, self._incidence_edges = self._data_loader.get_data()
        self.train_loader, self.val_loader, self.test_loader = self._data_loader.get_loaders(self.train_dataset, self.test_dataset, self._batch_size, 0.2)

        # Initialize model
        num_targets = len(self._symbols)
        num_features = len(config["cols"])
        self._model = Model(snapshots=self._hypergraphsnapshot, num_stock=num_targets, history_window=self.lookback_window,
                            num_feature=num_features, embedding_dim=self._hidden_dim,
                            rnn_hidden_unit=self._rnn_units, drop_prob=self._dropout)

        model_parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Build model, parameter size : {params}")

    def train(self):
        print("Begin training...")
        # print(f'1) {torch.cuda.memory_allocated()/1024.0/1024.0:.2f} MB')

        model = self._model
        if self._cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

        criterion = CustomMSELoss() # nn.MSELoss()
        epoch = 1
        epsilon = 1e-7
        min_loss = 1e+13
        epochs_no_improve = 0
        patience = self._earlystop
        results = {
            'train_loss': [],
            'val_loss': [],
        }

        # [ 훈련 ]
        try:
            while epoch <= self._epochs:
                model.train()
                train_loss = 0
                for X_batch, y_batch in self.train_loader:
                    if self._cuda:
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)  # * batch_size

                # [ 훈련데이터 오차 ]
                avg_train_loss = train_loss / len(self.train_loader.dataset)
                results['train_loss'].append(avg_train_loss)
                # [ 검증데이터 평가 ]
                model.eval()

                val_preds = []
                val_targets = []
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in self.val_loader:
                        if self._cuda:
                            X_batch = X_batch.cuda()
                            y_batch = y_batch.cuda()

                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)

                        val_preds.append(outputs.cpu())
                        val_targets.append(y_batch.cpu())

                avg_val_loss = val_loss / len(self.val_loader.dataset)
                results['val_loss'].append(avg_val_loss)

                val_preds = torch.cat(val_preds).numpy().flatten()
                val_gt = torch.cat(val_targets).numpy().flatten()
                val_mse = mean_squared_error(val_gt, val_preds)
                val_rmse = np.sqrt(val_mse)
                val_mae = mean_absolute_error(val_gt, val_preds)
                # val_mape = np.mean(np.abs((val_gt - val_preds) / (val_gt + epsilon))) * 100

                print(f"Epoch {epoch}: train_loss {avg_train_loss:.6f}, valid_loss {avg_val_loss:.6f}, "
                          f"valid_rmse {val_rmse:.6f}, valid_mae {val_mae:.6f}")

                if avg_val_loss < min_loss:
                    min_loss = avg_val_loss
                    epochs_no_improve = 0

                    # save model
                    current_time = datetime.now().strftime("%H%M%S")
                    torch.save(model, os.path.join('/home/dyd9800/artifact/models/estimate', f"model_best.pth"))
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print("Early stop due to no improvement")
                    break

                epoch += 1
        except KeyboardInterrupt:
            print("Training stopped by user")

        print("Done training!")
        with open('/home/dyd9800/artifact/models/estimate/train_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    def test(self):
        print("Begin testing...")
        if self._cuda:
            checkpoint = torch.load(os.path.join('/home/dyd9800/artifact/models/estimate', "model_best.pth"))
        else:
            checkpoint = torch.load(os.path.join('/home/dyd9800/artifact/models/estimate', "model_best.pth"), map_location='cpu')

        self._model = checkpoint
        model = self._model
        criterion = CustomMSELoss() #nn.MSELoss()

        # [ 테스트셋 평가 ]
        model.eval()

        test_preds = []
        test_targets = []
        test_loss = 0
        epsilon = 1e-7
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                if self._cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)

                test_preds.append(outputs.cpu())
                test_targets.append(y_batch.cpu())

        # [ 성능 평가 ]
        avg_test_loss = test_loss / len(self.test_loader.dataset)
        with open('/home/dyd9800/artifact/models/estimate/test_predicts.pkl', 'wb') as f:
            pickle.dump({'preds': test_preds, 'targets': test_targets}, f)

        test_preds = torch.cat(test_preds).numpy().flatten()
        test_targets = torch.cat(test_targets).numpy().flatten()

        test_mse = mean_squared_error(test_targets, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(test_targets, test_preds)
        # test_mape = np.mean(np.abs((test_targets - test_preds) / (test_targets + epsilon))) * 100

        print(f"Test avg_loss: {avg_test_loss}, RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")

        test_performance = {
            'rmse': test_rmse,
            'mae': test_mae,
        }
        return test_performance