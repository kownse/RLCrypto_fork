# -*- coding:utf-8 -*-
from models.Model import *
from models.ModelTrainer import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)
print()

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, b_dim, rnn_layers=1, dp=0.2, rnn_type='gru', linear_base=128):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.s_dim, linear_base, self.rnn_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.s_dim, linear_base, self.rnn_layers, batch_first=True)
        self.fc_s_1 = nn.Linear(linear_base, linear_base)
        self.fc_s_2 = nn.Linear(linear_base, linear_base // 2)
        self.fc_s_out = nn.Linear(linear_base // 2, 1)
        self.fc_pg_1 = nn.Linear(linear_base, linear_base)
        self.fc_pg_2 = nn.Linear(linear_base, linear_base // 2)
        self.fc_pg_out = nn.Linear(linear_base // 2, self.a_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax(dim=-1)
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, linear_base, dtype=torch.float32).cuda()
    
    def forward(self, state, hidden=None, train=False):
        state, h = self.rnn(state, hidden)
        if train:
            state = self.dropout(state)
        sn_out = self.relu(self.fc_s_1(state))
        sn_out = self.relu(self.fc_s_2(sn_out))
        if train:
            sn_out = self.dropout(sn_out)
        sn_out = self.fc_s_out(sn_out)
        
        pn_out = self.relu(self.fc_pg_1(state))
        pn_out = self.relu(self.fc_pg_2(pn_out))
        if train:
            pn_out = self.dropout(pn_out)
        pn_out = self.softmax(self.fc_pg_out(pn_out))
        if self.rnn_type == 'gru':
            return pn_out, sn_out, h.data
        elif self.rnn_type == 'lstm':
            return pn_out, sn_out, (h[0].data, h[1].data)

class RPG_Torch(Model):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learning_rate=1e-3,
                rnn_layers=1, normalize_length=10, rnn_type='gru', linear_base=128, drop=0.2):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.normalize_length = normalize_length
        self.pointer = 0
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        
        self.train_hidden = None
        self.trade_hidden = None
        self.actor = Actor(s_dim=self.s_dim, a_dim=self.a_dim, b_dim=self.b_dim, rnn_layers=rnn_layers, dp=drop, rnn_type=rnn_type, linear_base=linear_base)
        self.actor = self.actor.to(dev)

        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.trainer = ModelTrainer(self)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, _, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        if train:
            return torch.multinomial(a[:, 0, :], 1).cuda()
        else:
            return a[:, 0, :].argmax(dim=1)
    
    def _train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).transpose(0,1).cuda()
        s_next = torch.stack(self.s_next_buffer).transpose(0,1).cuda()
        r = torch.stack(self.r_buffer).transpose(0,1).cuda()
        a = torch.stack(self.a_buffer).transpose(0,1).cuda()
        a_hat, s_next_hat, self.train_hidden = self.actor(s, self.train_hidden, train=True)
        mse_loss = torch.nn.functional.mse_loss(s_next_hat, s_next)
        nll = -torch.log(a_hat.gather(2, a))
        pg_loss = (nll * r).mean()
        loss = mse_loss + pg_loss
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def reset_model(self):
        self.s_buffer = []
        self.a_buffer = []
        self.s_next_buffer = []
        self.r_buffer = []
        self.trade_hidden = None
        self.train_hidden = None
        self.pointer = 0
    
    def save_transition(self, state, action, reward, next_state):
        if self.pointer < self.batch_length:
            self.s_buffer.append(state)
            self.a_buffer.append(action)
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32).cuda())
            self.s_next_buffer.append(next_state)
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.a_buffer.pop(0)
            self.r_buffer.pop(0)
            self.s_next_buffer.pop(0)
            self.s_buffer.append(state)
            self.a_buffer.append(action)
            self.r_buffer.append(torch.tensor(reward[:, None], dtype=torch.float32).cuda())
            self.s_next_buffer.append(next_state)
    
    def load_model(self, model_path='./RPG_Torch'):
        self.trainer.load_model(model_path)
    
    def save_model(self, model_path='./RPG_Torch'):
        self.trainer.save_model(model_path)
    
    def train(self, asset_data, c, train_length, epoch=0):
        self.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        train_reward = []
        train_actions = []
        for t in range(self.normalize_length, train_length):
            state = self.trainer.check_cache_get_state(asset_data, t, self.normalize_length, self.trainer.all_train_states)
            next_state = asset_data[:, :, 'diff'].iloc[t].values
            next_state = torch.tensor(next_state)[:, None].cuda()
            action = self._trade(state, train=True)
            action_np = action.cpu().numpy().flatten()
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np - c * np.abs(previous_action - action_np)
            self.save_transition(state=state, action=action, next_state=next_state, reward=r)
            train_reward.append(r)
            train_actions.append(action_np)
            previous_action = action_np
            if t % self.batch_length == 0:
                self._train()
        self.reset_model()
        print(epoch, 'train_reward', np.sum(np.mean(train_reward, axis=1)), np.mean(train_reward))
        return train_reward, train_actions
    
    def back_test(self, asset_data, c, test_length, epoch=0):
        self.reset_model()
        previous_action = np.zeros(asset_data.shape[0])
        test_reward = []
        test_actions = []
        for t in range(asset_data.shape[1] - test_length, asset_data.shape[1]):
            state = self.trainer.check_cache_get_state(asset_data, t, asset_data.shape[1] - test_length, self.trainer.all_train_states)
            action = self._trade(state=state, train=False)
            action_np = action.cpu().numpy().flatten()
            r = asset_data[:, :, 'diff'].iloc[t].values * action_np - c * np.abs(previous_action - action_np)
            test_reward.append(r)
            test_actions.append(action_np)
            previous_action = action_np
        self.reset_model()
        print(epoch, 'backtest reward', np.sum(np.mean(test_reward, axis=1)), np.mean(test_reward))
        return test_reward, test_actions
    
    def trade(self, asset_data):
        action_np = self.trainer.train(asset_data)
        return action_np/(np.sum(action_np)+1e-10)
    
    @staticmethod
    def create_new_model(asset_data,
                         c,
                         normalize_length,
                         batch_length,
                         train_length,
                         max_epoch,
                         learning_rate,
                         pass_threshold,
                         model_path):
        return ModelTrainer.create_new_model(RPG_Torch,
                                            asset_data,
                                            c,
                                            normalize_length,
                                            batch_length,
                                            train_length,
                                            max_epoch,
                                            learning_rate,
                                            pass_threshold,
                                            model_path)
