# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:26:55 2018

@author: Administrator
"""
import time
start_all=time.time()
from math import sqrt
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
train_y = train_y.reshape((train_y.shape[0],1))
test_y = test_y.reshape((test_y.shape[0],1))
class Dataset(object):
    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass
    def data(self):
        return self._data
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples 
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]
train_XX=Dataset(train_X)
train_yy=Dataset(train_y)
#device = torch.device("cpu")
hidden_num=50
layer_num=1
time_step=1
input_num=8
output_num=1
epoch_num=50
batch_size=72
learning_rate=0.001
class Net(nn.Module):
    def __init__(self,input_num,hidden_num,layer_num,output_num):
        super(Net, self).__init__()
        self.lstm1=nn.LSTM(input_size=input_num,hidden_size=hidden_num,num_layers=layer_num)
        self.fc1=nn.Linear(hidden_num,output_num)
    def forward(self,x):
        out,_=self.lstm1(x)
        x=self.fc1(out[:,-1,:])
        x=x.view(-1,output_num)
        return x
model=Net(input_num,hidden_num,layer_num,output_num)
#model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_epoch=[0]*epoch_num
def train():
    model.train()
    xx=0
    n=0
    while train_XX._epochs_completed<epoch_num:
        batch_X=train_XX.next_batch(batch_size)
        batch_y=train_yy.next_batch(batch_size)
        batch_X=torch.from_numpy(batch_X)
        batch_y=torch.from_numpy(batch_y)
        optimizer.zero_grad()
        output=model(batch_X)
        loss=F.l1_loss(output,batch_y)
        loss.backward()
        optimizer.step()
        if train_XX._epochs_completed==xx:
            n=n+1
            loss_epoch[xx]+=float(loss)
        else:
            loss_epoch[xx]=loss_epoch[xx]/n
            xx=train_XX._epochs_completed
            n=1
            print('epoch%d:mae=%f' % (xx-1,loss_epoch[xx-1]))
            if xx!=epoch_num:
                loss_epoch[xx]+=float(loss)
start_train=time.time()
train()       
end_train=time.time()
time_train=end_train-start_train
print("The time cost in train:%f" % time_train)
def test():
    model.eval()
    with torch.no_grad():
        test_XX=torch.from_numpy(test_X)
        yhat=model(test_XX)
    return yhat
yhat=test()
end_all=time.time()
time_all=end_all-start_all
print("All the time cost:%f" % time_all)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %f' % rmse)