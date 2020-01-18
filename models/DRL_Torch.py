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
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.s_dim, linear_base, self.rnn_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.s_dim, linear_base, self.rnn_layers, batch_first=True)
        self.fc_policy_1 = nn.Linear(linear_base, linear_base)
        self.fc_policy_2 = nn.Linear(linear_base, linear_base // 2)
        self.fc_policy_out = nn.Linear(linear_base // 2, a_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax()
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, linear_base, dtype=torch.float32).cuda()
    
    def forward(self, state, hidden=None, train=False):
        state, h = self.rnn(state, hidden)
        if train:
            state = self.dropout(state)

        state = self.relu(self.fc_policy_1(state))
        state = self.relu(self.fc_policy_2(state))
        action = self.softmax(self.fc_policy_out(state).squeeze())
        if isinstance(self.rnn, nn.GRU):
            return action, h.data
        elif isinstance(self.rnn, nn.LSTM):
            return action, (h[0].data, h[1].data)


class DRL_Torch(Model):
    def __init__(self, s_dim, a_dim, b_dim, batch_length=64, learning_rate=1e-3,
                rnn_layers=1, normalize_length=10, rnn_type='gru', linear_base=128, drop=0.2, opt_type='adam'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.normalize_length = normalize_length
        self.pointer = 0
        self.s_buffer = []
        self.d_buffer = []
        self.rnn_type = rnn_type
        
        self.train_hidden = None
        self.trade_hidden = None
        self.actor = Actor(s_dim=self.s_dim, a_dim=self.a_dim, b_dim=self.b_dim, rnn_layers=rnn_layers, dp=drop, rnn_type=rnn_type, linear_base=linear_base)
        self.actor = self.actor.to(dev)
        print(opt_type)
        if opt_type is not None:
            if opt_type == 'adam':
                self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
            elif opt_type == 'adamax':
                self.optimizer = optim.Adamax(self.actor.parameters(), lr=learning_rate)
            else:
                print('not support yet')
                exit(1)
        self.trainer = ModelTrainer(self)
    
    def _trade(self, state, train=False):
        with torch.no_grad():
            a, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        return a
    
    def _train(self):
        self.optimizer.zero_grad()
        s = torch.stack(self.s_buffer).transpose(0,1).cuda()
        d = torch.stack(self.d_buffer).cuda()
        a_hat, self.train_hidden = self.actor(s, self.train_hidden, train=True)
        reward = -(a_hat[:, :-1] * d).mean()
        reward.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def reset_model(self):
        self.s_buffer = []
        self.d_buffer = []
        self.trade_hidden = None
        self.train_hidden = None
        self.pointer = 0
    
    def save_transition(self, state, reward):
        if self.pointer < self.batch_length:
            self.s_buffer.append(state)
            self.d_buffer.append(torch.tensor(reward, dtype=torch.float32).cuda())
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.d_buffer.pop(0)
            self.s_buffer.append(state)
            self.d_buffer.append(torch.tensor(reward, dtype=torch.float32).cuda())
    
    def load_model(self, model_path='./DRL_Torch'):
        self.trainer.load_model(model_path)
    
    def save_model(self, model_path='./DRL_Torch'):
        self.trainer.save_model(model_path)

    def train(self, asset_data, c, train_length, epoch=0):
        return self.trainer.train(asset_data, c, train_length, epoch)
    
    def back_test(self, asset_data, c, test_length, epoch=0):
        return self.trainer.back_test(asset_data, c, test_length, epoch)
    
    def trade(self, asset_data):
        action_np = self.trainer.trade(asset_data)
        return action_np[:-1]
