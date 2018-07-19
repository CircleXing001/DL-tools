import mxnet as mx
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)
#i use this model on different tools,so i use the same data source samply
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels  
x_train = x_train.reshape(-1,1,28, 28).astype('float32')  
x_test = x_test.reshape(-1,1,28, 28).astype('float32')
def get_data_iter(batch_size):  
    """ 
    create data iterator with NDArrayIter 
    """  
    train = mx.io.NDArrayIter(x_train,y_train,batch_size)  
    val = mx.io.NDArrayIter(x_test,y_test,batch_size)
    return train, val 
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
fc7=mx.symbol.softmax(fc6)
out=mx.symbol.MakeLoss(-mx.symbol.sum(label*mx.symbol.log(fc7+1e-10),axis=1))
model = mx.mod.Module(symbol=out,context=mx.cpu())
model.bind(data_shapes=get_data_iter(50)[0].provide_data,label_shapes=get_data_iter(50)[0].provide_label)
model.init_params()
def check_label_shapes(labels, preds, wrap=False, shape=False):
   if not shape:
       label_shape, pred_shape = len(labels), len(preds)
   else:
       label_shape, pred_shape = labels.shape, preds.shape

   if label_shape != pred_shape:
       raise ValueError("Shape of labels {} does not match shape of "
                        "predictions {}".format(label_shape, pred_shape))

   if wrap:
       if isinstance(labels, ndarray.ndarray.NDArray):
           labels = [labels]
       if isinstance(preds, ndarray.ndarray.NDArray):
           preds = [preds]

   return labels, preds
class My_Accuracy(mx.metric.EvalMetric):

   def __init__(self, axis=1, name='accuracy',
                output_names=None, label_names=None):
       super(My_Accuracy, self).__init__(
           name, axis=axis,
           output_names=output_names, label_names=label_names)
       self.axis = axis

   def update(self, labels, preds):
       for label, pred_label in zip(labels, preds):
           
           pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
           pred_label = pred_label.asnumpy().astype('int32')
           label=mx.ndarray.argmax(label,axis=self.axis)
           label = label.asnumpy().astype('int32')

           labels, preds = check_label_shapes(label, pred_label)

           self.sum_metric += (pred_label.flat == label.flat).sum()
           self.num_inst += len(pred_label.flat)

eval_metrics = mx.metric.CompositeEvalMetric()
eval_metrics.add(My_Accuracy())			
model.fit(
    train_data =get_data_iter(50)[0],
    eval_data = get_data_iter(50)[1], 
	eval_metric=eval_metrics,
    optimizer = 'sgd',
	optimizer_params={'learning_rate':0.01},
    num_epoch=10,
    batch_end_callback=mx.callback.Speedometer(50)
    )