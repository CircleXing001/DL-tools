import time
start_all=time.time()
from pandas import read_csv
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import numpy as np
from math import sqrt
#数据预处理：标准化并转化为监督学习形式
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
#分割训练集和测试集
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

def get_data_iter(batch_size):  
    """ 
    create data iterator with NDArrayIter 
    """  
    train = mx.io.NDArrayIter(  
        train_X, train_y, batch_size,label_name='label')  
    val = mx.io.NDArrayIter(  
        test_X, test_y, batch_size,label_name='label')  
    return train, val  

#design network
import mxnet as mx
import logging
logging.basicConfig(level=logging.DEBUG)
hidden_num=50
num_layers=1
time_step=1
input_num=8
output_num=1
epoch_num=50
batch_size=72
learning_rate=0.001
gpu_device=mx.cpu()
with mx.Context(gpu_device):
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=hidden_num, prefix='lstm_l%d_'%i))
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    stack.reset()
    outputs, states = stack.unroll(time_step,inputs=data, merge_outputs=True)
    out_lstm = mx.sym.Reshape(outputs, shape=(-1, hidden_num))
    out = mx.symbol.FullyConnected(data=outputs,num_hidden=output_num,name='fc')
    out = mx.sym.MAERegressionOutput(data=out,label=label)

    model = mx.mod.Module(symbol=out,data_names=('data',),label_names=('label',),context=mx.cpu())
    model.bind(data_shapes=get_data_iter(batch_size)[0].provide_data,label_shapes=get_data_iter(batch_size)[0].provide_label)
    model.init_params()
    start_train=time.time()
    x=model.fit(
            train_data =get_data_iter(batch_size)[0],
            #    eval_data = get_data_iter(batch_size)[1],
            eval_metric='mae', 
            optimizer = 'adam',
            optimizer_params={'learning_rate':learning_rate},
            num_epoch=epoch_num,
            batch_end_callback=mx.callback.Speedometer(batch_size)
            )
    end_train=time.time()
    time_train=end_train-start_train
    print("The time cost in train:%f" % time_train)
    yhat = model.predict(mx.io.NDArrayIter(test_X,np.zeros(test_X.shape[0]),test_X.shape[0]),reset=False)
end_all=time.time()
time_all=end_all-start_all
print("All the time cost:%f" % time_all)
yhat=yhat.asnumpy()
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 1:]),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %f' % rmse)