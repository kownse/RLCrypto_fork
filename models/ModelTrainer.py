from models.Model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os

class ModelTrainer:
    def __init__(self, owner):
        self.owner = owner
        self.all_train_states = []
        self.all_test_states = []

    def load_model(self, model_path='./DRL_Torch'):
        self.owner.actor = torch.load(model_path + '/model.pkl')
    
    def save_model(self, model_path='./DRL_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.owner.actor, model_path + '/model.pkl')

    def check_cache_get_state(self, asset_data, t, start, cache):
        idx = t - start
        if idx <= (len(cache) - 1):
            state = cache[idx]
        else:
            data = asset_data.iloc[:, t - self.owner.normalize_length:t, :].values
            # this is actually zscore
            state = ((data[:,-1,:] - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:,-1,:]
            state = torch.tensor(state).cuda()
            cache.append(state)
        return state

    @abstractmethod
    def reset_model(self, model_path):
        pass

    def train(self, asset_data, c, train_length, epoch=0):
        self.owner.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        train_reward = []
        train_actions = []
        for t in range(self.owner.normalize_length, train_length):
            state = self.check_cache_get_state(asset_data, t, self.owner.normalize_length, self.all_train_states)

            action = self.owner._trade(state, train=True)
            action_np = action.cpu().numpy().flatten()
            action_np = np.round(action_np, decimals=1)
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np[:-1] - c * np.abs(previous_action - action_np[:-1])
            self.owner.save_transition(state=state, reward=asset_data[:, :, 'diff'].iloc[t].values)
            train_reward.append(r)
            train_actions.append(action_np)
            previous_action = action_np[:-1]
            if t % self.owner.batch_length == 0:
                self.owner._train()
        self.owner.reset_model()
        print(epoch, 'train_reward', np.sum(np.sum(train_reward, axis=1)), np.mean(train_reward))
        return train_reward, train_actions

    def back_test(self, asset_data, fee, test_length, epoch=0):
        self.owner.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        test_reward = []
        test_actions = []
        start = asset_data.shape[1] - test_length
        for t in range(start, asset_data.shape[1]):
            state = self.check_cache_get_state(asset_data, t, start, self.all_test_states)

            action = self.owner._trade(state=state, train=False)
            action_np = action.cpu().numpy().flatten()
            action_np = np.round(action_np, decimals=1)
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np[:-1] - fee * np.abs(previous_action - action_np[:-1])
            test_reward.append(r)
            test_actions.append(action_np)
            previous_action = action_np[:-1]
        self.owner.reset_model()
        print(epoch, 'backtest reward', np.sum(np.sum(test_reward, axis=1)), np.mean(test_reward))
        return test_reward, test_actions

    def trade(self, asset_data):
        if self.owner.trade_hidden is None:
            self.owner.reset_model()
            action_np = np.zeros(asset_data.shape[0])
            for t in range(asset_data.shape[1] - self.owner.batch_length, asset_data.shape[1]):
                data = asset_data.iloc[:, t - self.owner.normalize_length + 1:t + 1, :].values
                state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
                state = torch.tensor(state).cuda()
                action = self.owner._trade(state=state, train=False)
                action_np = action.cpu().numpy().flatten()
        else:
            data = asset_data.iloc[:, -self.owner.normalize_length:, :].values
            state = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-5))[:, -1, :]
            state = torch.tensor(state).cuda()
            action = self.owner._trade(state=state, train=False)
            action_np = action.cpu().numpy().flatten()
        return action_np

    @staticmethod
    def create_new_model(ModelClass,
                         asset_data,
                         c,
                         normalize_length,
                         rnn_layers,
                         rnn_type,
                         linear_base,
                         batch_length,
                         train_length,
                         max_epoch,
                         learning_rate,
                         model_path,
                         drop=0.2,
                         patient=10,
                         patient_rounds=3,
                         data_interval='1h'):
        torch.cuda.empty_cache()
        round = 0
        current_model_reward = -np.inf
        best_model_reward = -np.inf
        model_path = 'models_train/%s_%s_%d%s_base%d_drop%.2f_normlen%d' % (model_path, data_interval, rnn_layers, rnn_type, linear_base, drop, normalize_length)
        best_model_path = model_path + '_best'
        model = None
        while round < patient_rounds:
            round = round + 1
            unbreak_epoch = 0
            model = ModelClass(s_dim=asset_data.shape[2],
                              a_dim=asset_data.shape[0] + 1,
                              b_dim=asset_data.shape[0],
                              batch_length=batch_length,
                              learning_rate=learning_rate,
                              rnn_layers=rnn_layers,
                              normalize_length=normalize_length,
                              rnn_type=rnn_type,
                              linear_base=linear_base,
                              drop=drop)
            model.reset_model()
            lr_scheduler = ReduceLROnPlateau(model.optimizer, mode='max', factor=0.5, patience=6, verbose=True)

            for e in range(max_epoch):
                train_reward, train_actions = model.train(asset_data, c=c, train_length=train_length, epoch=e)
                test_reward, test_actions = model.back_test(asset_data, c=c, test_length=asset_data.shape[1] - train_length)
                current_model_reward = np.sum(np.sum(test_reward, axis=1))
                lr_scheduler.step(current_model_reward)

                if current_model_reward > best_model_reward:
                    best_model_reward = current_model_reward
                    model.save_model('%s_%.2f' % (best_model_path, current_model_reward))
                    print('save best model for current_reward:', current_model_reward, 'to', best_model_path)
                    unbreak_epoch = 0
                else:
                    unbreak_epoch = unbreak_epoch + 1
                    if unbreak_epoch >patient:
                        print("No more patient")
                        break

                break
        print('model created successfully, backtest reward:', current_model_reward)
        model.save_model(model_path)
        return model