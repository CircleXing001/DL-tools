WORD_NUM = 1427
SENTENCE_LENGTH = 27
IMG_NUM = 1449
IMG_WIDTH = 50
IMG_HEIGHT = 50
CHANNEL_NUM = 3
QA_PATH = 'qa.894.raw.txt'
IMG_PATH = 'the_new'
BATCH_SIZE = 100
EPOCH_NUM = 5
OPTIM = 'adam' # 'rmsprop'

import time
import os
import gc
from PIL import Image
import numpy as np
import mxnet as mx 
from mxnet import gluon,nd 
from mxnet.gluon import nn,rnn

def load_data():
    print('loading data...')
    ques=[]
    answ=[]
    no_qa=0
    no_word=1
    words={}
    fo=open(QA_PATH,'r')
    qa=fo.readlines()
    for i in qa:
        if no_qa%2==0:
            ques.append(i[:-3])
            w=i[:-3].split(' ')[:-3]
            for word in w:
                if word in words:
                    pass
                else:
                    words[word]=no_word
                    no_word+=1
        else:
            if len(i.split(','))>1:
                ques.pop()
            else:
                answ.append(i[:-1])
                if i[:-1] in words:
                    pass
                else:
                    words[i[:-1]]=no_word
                    no_word+=1
        no_qa+=1
    fo.close()
    question=np.zeros((len(ques),SENTENCE_LENGTH,))
    answer=np.zeros((len(answ),WORD_NUM))
    nn=0
    for q in ques:
        nnn=0
        for word in q.split(' ')[:-3]:
            question[nn,nnn,]=words[word]
            nnn+=1
        nn+=1
    nn=0
    for a in answ:
        answer[nn,words[a]]=1
        nn+=1
    images=np.empty((IMG_NUM,IMG_WIDTH,IMG_HEIGHT,CHANNEL_NUM),dtype='float32')
    data_image=np.empty((len(ques),IMG_WIDTH,IMG_HEIGHT,CHANNEL_NUM),dtype='float32')
    for i in range(IMG_NUM):
        i=i+1
        img=Image.open(IMG_PATH+'/image'+str(i)+'.png')
        arr = np.asarray(img,dtype='float32')
        images[i-1,:,:,:]=arr
    nn=0
    for q in ques:
        number=int(q.split('image')[-1])
        data_image[nn,:,:,:]=images[number-1]
        nn+=1
    print('have got the data!')
    del ques,answ,words,images
    gc.collect()
    return question,data_image,answer
data=load_data()
im=data[1]
im=im.reshape((-1,CHANNEL_NUM,IMG_WIDTH,IMG_HEIGHT))
qq=data[0]
aa=data[2]

ctx = mx.gpu()
class Net(nn.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1=nn.Conv2D(channels=64,kernel_size=(3,3),padding=(1,1),in_channels=3,activation='relu')
        self.conv2=nn.Conv2D(channels=64,kernel_size=(3,3),in_channels=64,activation='relu')
        self.pool1=nn.MaxPool2D((2,2))
        self.conv3=nn.Conv2D(channels=128,kernel_size=(3,3),padding=(1,1),in_channels=64,activation='relu')
        self.conv4=nn.Conv2D(channels=128,kernel_size=(3,3),in_channels=128,activation='relu')
        self.pool2=nn.MaxPool2D((2,2))
        self.conv5=nn.Conv2D(channels=256,kernel_size=(3,3),padding=(1,1),in_channels=128,activation='relu')
        self.conv6=nn.Conv2D(channels=256,kernel_size=(3,3),in_channels=256,activation='relu')
        self.conv7=nn.Conv2D(channels=256,kernel_size=(3,3),in_channels=256,activation='relu')
        self.pool3=nn.MaxPool2D((2,2))
        self.fc1=nn.Flatten()
        self.fc2=nn.Embedding(WORD_NUM,256)
        self.lstm1=rnn.LSTM(256)
        self.fc3=nn.Dense(WORD_NUM)

    def forward(self, x, y):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.pool1(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.pool2(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.pool3(x)
        x=self.fc1(x)
        y=self.fc2(y)
        y=self.lstm1(y)
        y=y[:,-1,:]
        output=nd.Concat(x,y,dim=1)
        output=self.fc3(output)
        output=nd.softmax(output)
        return output
model=Net()
model.initialize(mx.initializer.Xavier(),ctx=ctx)
trainer = gluon.Trainer(model.collect_params(), OPTIM,{'learning_rate': 0.001})
def train():
	completed_epoch=0
	start=0
	epoch_loss=0
	epoch_accuracy=0
	no=0
	while completed_epoch<EPOCH_NUM:
		if start+BATCH_SIZE<=im.shape[0]:
			image_input=nd.array(im[start:start+BATCH_SIZE],ctx=ctx)
			question_input=nd.array(qq[start:start+BATCH_SIZE],ctx=ctx)
			answer_output=nd.array(aa[start:start+BATCH_SIZE],ctx=ctx)
			with mx.autograd.record():
				output=model(image_input,question_input)
				loss=-nd.sum(answer_output*nd.log(output))
			loss.backward()
			trainer.step(1)
			output1=output.asnumpy()
			answer_output1=answer_output.asnumpy()
			loss1=loss.asnumpy()
			accuracy=np.equal(np.argmax(output1,1), np.argmax(answer_output1,1))
			accuracy=accuracy.astype('float')
			accuracy=np.mean(accuracy)
			epoch_loss+=loss1
			epoch_accuracy+=accuracy
			start=start+BATCH_SIZE
			no+=1
		else:
			image_input=nd.array(np.concatenate((im[start:],im[:BATCH_SIZE-im.shape[0]+start]),0),ctx=ctx)
			question_input=nd.array(np.concatenate((qq[start:],qq[:BATCH_SIZE-qq.shape[0]+start]),0),ctx=ctx)
			answer_output=nd.array(np.concatenate((aa[start:],aa[:BATCH_SIZE-aa.shape[0]+start]),0),ctx=ctx)
			with mx.autograd.record():
				output=model(image_input,question_input)
				loss=-nd.sum(answer_output*nd.log(output))
			loss.backward()
			trainer.step(1)
			completed_epoch+=1
			start=BATCH_SIZE-im.shape[0]+start
			output1=output.asnumpy()
			answer_output1=answer_output.asnumpy()
			loss1=loss.asnumpy()
			accuracy=np.equal(np.argmax(output1,1), np.argmax(answer_output1,1))
			accuracy=accuracy.astype('float')
			accuracy=np.mean(accuracy)
			epoch_loss+=loss1
			epoch_accuracy+=accuracy
			no+=1
			epoch_loss=epoch_loss/no
			epoch_accuracy=epoch_accuracy/no
			print('epoch%d:loss-%f,accuracy-%f'%(completed_epoch,epoch_loss,epoch_accuracy))
			epoch_loss=0
			epoch_accuracy=0
			no=0
start=time.time()
train()
end=time.time()
train_time=end-start
print('The time cost in train:%f' %train_time)
