import time 
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
   
x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.train.labels  
x_train = x_train.reshape(-1, 28, 28,1).astype('float32')  
x_test = x_test.reshape(-1,28, 28,1).astype('float32') 

def net_model():
    model=Sequential()
    model.add(Convolution2D(32,(5,5),padding='same',  
                            input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same'))
    model.add(Convolution2D(64,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2),padding='same'))
    model.add(Reshape((7*7*64,)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=["accuracy"])
    return model
model=net_model()
train_start=time.time()
model.fit(x_train,y_train,batch_size=50,epochs=10,verbose=1)
train_end=time.time()
train_time=train_end-train_start
print('The time cost in train:%f'%train_time)
