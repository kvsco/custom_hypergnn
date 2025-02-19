import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self._dropout = 0.1
        self._batch_size = 16
        self._epochs = 20
        self._cuda = True  # in gpu server
        self._earlystop = 10
        self._eval_iter = 10
        self._rnn_units = 16

        # Get data
        self.lookback_window = config['lookback_window']
        self.lookahead_window = config['lookahead_window']
        self._data_loader = MyDataLoader(self._config)
        self.train_dataset, self._hypergraphsnapshot, self._incidence_edges = self._data_loader.get_data()
        self.train_loader, self.val_loader, self.test_loader = self._data_loader.get_loaders(self.train_dataset, self._batch_size)

        # Initialize model
        num_targets = len(self._symbols)
        num_features = len(config["cols"])
        self._model = Model(snapshots=self._hypergraphsnapshot, num_stock=num_targets,
                            history_window=self.lookback_window, lookahead_window=self.lookahead_window,
                            num_feature=num_features, embedding_dim=self._hidden_dim, rnn_hidden_unit=self._rnn_units, drop_prob=self._dropout)

        model_parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Build model, parameter size : {params}")

    def train(self, setting):
        print("Begin training...")
        # print(f'1) {torch.cuda.memory_allocated()/1024.0/1024.0:.2f} MB')

        model = self._model
        if self._cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

        criterion = nn.MSELoss() #CustomMSELoss()
        epoch = 1
        epsilon = 1e-7
        min_loss = 1e+13
        epochs_no_improve = 0
        patience = 3
        results = {
            'train_loss': [],
            'val_loss': [],
        }
        time_now = time.time()
        train_steps = len(self.train_loader)
        # [ 훈련 ]
        try:
            while epoch <= self._epochs:
                iter_count = 0
                epoch_time = time.time()
                model.train()
                # train_loss = 0
                train_loss = []
                for i, (X_batch, y_batch) in enumerate(self.train_loader):
                    iter_count += 1
                    optimizer.zero_grad()

                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                    outputs = model(X_batch) # (16, 48, 60)
                    outputs = outputs[:, :, :30] # (16, 48, 30)
                    pred = outputs[:, 0, :]
                    true = y_batch[:, 0, :]
                    loss = criterion(pred, true)

                    loss.backward()
                    optimizer.step()
                    # train_loss += loss.item() * X_batch.size(0)  # * batch_size
                    train_loss.append(loss.item())
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self._epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                print("Epoch: {} cost time: {}".format(epoch, time.time() - epoch_time))

                # [ 훈련데이터 오차 ]
                train_loss = np.average(train_loss)
                results['train_loss'].append(train_loss)
                # [ 검증데이터 평가 ]
                model.eval()

                val_preds = []
                val_targets = []
                # val_loss = 0
                val_loss = []
                with torch.no_grad():
                    for X_batch, y_batch in self.val_loader:
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()

                        outputs = model(X_batch)
                        outputs = outputs[:, :, :30]
                        loss = criterion(outputs[:, 0, :], y_batch[:, 0, :])
                        # val_loss += loss.item() * X_batch.size(0)
                        val_loss.append(loss.item())
                        val_preds.append(outputs.cpu())
                        val_targets.append(y_batch.cpu())

                val_loss = np.average(val_loss)
                # avg_val_loss = val_loss / len(self.val_loader.dataset)
                results['val_loss'].append(val_loss)

                val_preds = torch.cat(val_preds).numpy().flatten()
                val_gt = torch.cat(val_targets).numpy().flatten()
                val_mse = mean_squared_error(val_gt, val_preds)
                val_rmse = np.sqrt(val_mse)
                val_mae = mean_absolute_error(val_gt, val_preds)

                # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, val_loss))
                print(f"Epoch {epoch}: train_loss {train_loss:.6f}, valid_loss {val_loss:.6f}, "
                          f"valid_rmse {val_rmse:.6f}, valid_mae {val_mae:.6f}")

                if val_loss < min_loss:
                    min_loss = val_loss
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

    def test(self, setting):
        print("Begin testing...")
        if self._cuda:
            checkpoint = torch.load(os.path.join('/home/dyd9800/artifact/models/estimate', "model_best.pth"))
        else:
            checkpoint = torch.load(os.path.join('/home/dyd9800/artifact/models/estimate', "model_best.pth"), map_location='cpu')

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self._model = checkpoint
        model = self._model
        criterion = nn.MSELoss() #CustomMSELoss()

        # [ 테스트셋 평가 ]
        model.eval()

        preds = []
        trues = []
        inputx = []
        test_loss = 0
        epsilon = 1e-7
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(self.test_loader):
                if self._cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                outputs = model(X_batch)
                outputs = outputs[:, :, :30]
                loss = criterion(outputs[:, 0, :], y_batch[:, 0, :])

                pred = outputs[:, 0, :].detach().cpu().numpy()
                true = y_batch[:, 0, :].detach().cpu().numpy()

                # test_loss += loss.item() * X_batch.size(0)

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = X_batch.detach().cpu().numpy() # input[0, 0, :, -3]
                    gt = np.concatenate((input[14, 0, :, -3], true[14, :]), axis=0) # -3 : tps
                    pd = np.concatenate((input[14, 0, :, -3], pred[14, :]), axis=0)
                    self.visual(gt, pd, '3701_tps', os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # [ 성능 평가 ]
        mae, mse, rmse, mape, mspe, rse = self.metric(preds, trues)
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def visual(self, true, preds=None, target='feature', name='./pic/test.pdf'):
        """
        Results visualization
        """
        plt.figure()
        plt.plot(preds, label='Prediction', linewidth=2)
        plt.plot(true, label='GroundTruth', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title(target)

        plt.legend()
        plt.savefig(name, bbox_inches='tight')
        plt.close()

    def metric(self, pred, true):
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred - true) / true))
        mspe = np.mean(np.square((pred - true) / true))
        rse = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

        return mae, mse, rmse, mape, mspe, rse