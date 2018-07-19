import mxnet as mx
import time
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
   
x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels  
x_train = x_train.reshape(-1,1,28, 28).astype('float32')  
x_test = x_test.reshape(-1,1,28, 28).astype('float32')
y_train=np.argmax(y_train,axis=1)
y_test=np.argmax(y_test,axis=1)
def get_data_iter(batch_size):  
    """ 
    create data iterator with NDArrayIter 
    """  
    train = mx.io.NDArrayIter(x_train,y_train,batch_size)  
    val = mx.io.NDArrayIter(x_test,y_test,batch_size)
    return train, val 
gpu_device=mx.cpu()
with mx.Context(gpu_device):
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('softmax_label')
	conv1=mx.symbol.Convolution(data,kernel=(5,5),stride=(1,1),pad=(2,2),num_filter=32)
	fc1=mx.symbol.Activation(conv1,act_type='relu')
	pool1=mx.symbol.Pooling(fc1,kernel=(2,2),pool_type='max',stride=(2,2))
	conv2=mx.symbol.Convolution(pool1,kernel=(5,5),stride=(1,1),pad=(2,2),num_filter=64)
	fc2=mx.symbol.Activation(conv2,act_type='relu')
	pool2=mx.symbol.Pooling(fc2,kernel=(2,2),pool_type='max',stride=(2,2))
	fc3=mx.sym.Reshape(pool2,shape=(-1,7*7*64))
	fc4=mx.symbol.FullyConnected(fc3,num_hidden=1024)
	fc5=mx.symbol.Activation(fc4,act_type='relu')
	drop1=mx.symbol.Dropout(fc5,p=0.5)
	fc6=mx.symbol.FullyConnected(drop1,num_hidden=10)
	out=mx.symbol.SoftmaxOutput(fc6,label)
	model = mx.mod.Module(symbol=out,context=mx.cpu())
	model.bind(data_shapes=get_data_iter(50)[0].provide_data,label_shapes=get_data_iter(50)[0].provide_label)
	model.init_params()
	train_start=time.time()			
	model.fit(
		train_data =get_data_iter(50)[0],
		eval_data = get_data_iter(50)[1], 
		eval_metric='acc',
		optimizer = 'sgd',
		optimizer_params={'learning_rate':0.01},
		num_epoch=10,
		batch_end_callback=mx.callback.Speedometer(50)
		)
	train_end=time.time()
	train_time=train_end-train_start
	print('The time cost in train:%f'%train_time) 
