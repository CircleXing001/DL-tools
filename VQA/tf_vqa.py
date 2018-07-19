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
import tensorflow as tf
from tensorflow.contrib import rnn

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
qq=data[0]
aa=data[2]

with tf.device('/gpu:0'):
	image_input = tf.placeholder('float32',[None,IMG_WIDTH,IMG_HEIGHT,CHANNEL_NUM])
	question_input = tf.placeholder('int32',[None,SENTENCE_LENGTH])
	answer_output = tf.placeholder('float32',[None,WORD_NUM])
	conv1=tf.layers.conv2d(image_input,64,(3,3),padding = 'same',activation = tf.nn.relu)
	conv2=tf.layers.conv2d(conv1,64,(3,3),activation =tf.nn.relu)
	pool1=tf.layers.max_pooling2d(conv2,(2,2),(2,2))
	conv3=tf.layers.conv2d(pool1,128,(3,3),padding = 'same',activation = tf.nn.relu)
	conv4=tf.layers.conv2d(conv3,128,(3,3),activation =tf.nn.relu)
	pool2=tf.layers.max_pooling2d(conv4,(2,2),(2,2))
	conv5=tf.layers.conv2d(pool2,256,(3,3),padding = 'same',activation = tf.nn.relu)
	conv6=tf.layers.conv2d(conv5,256,(3,3),activation =tf.nn.relu)
	conv7=tf.layers.conv2d(conv6,256,(3,3),activation =tf.nn.relu)
	pool3=tf.layers.max_pooling2d(conv7,(2,2),(2,2))
	encoded_image=tf.contrib.layers.flatten(pool3)
	
	embedded_question=tf.contrib.layers.embed_sequence(question_input,WORD_NUM,256)
	lstm_cell=rnn.BasicLSTMCell(256)
	outputs,_=tf.nn.dynamic_rnn(lstm_cell,embedded_question,dtype=tf.float32)
	encoded_question=outputs[:,-1,:]
	merged=tf.concat([encoded_image,encoded_question],1)
	output=tf.layers.dense(merged,WORD_NUM,activation =tf.nn.softmax)
	loss=-tf.reduce_sum(answer_output*tf.log(output))
	train_step = tf.train.AdamOptimizer().minimize(loss)
	correct_prediction = tf.equal(tf.argmax(answer_output,1), tf.argmax(output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	init=tf.global_variables_initializer()
start=time.time()
print('go')
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	sess.run(init)
	completed_epoch=0
	start=0
	epoch_loss=0
	epoch_accuracy=0
	no=0
	while completed_epoch<EPOCH_NUM:
		if start+BATCH_SIZE<=im.shape[0]:
			_,batch_loss,batch_accuracy=sess.run([train_step,loss,accuracy],feed_dict={image_input:im[start:start+BATCH_SIZE],question_input:qq[start:start+BATCH_SIZE],answer_output:aa[start:start+BATCH_SIZE]})
			start=start+BATCH_SIZE
			epoch_loss+=batch_loss
			epoch_accuracy+=batch_accuracy
			no+=1
			print(no)
		else:
			_,batch_loss,batch_accuracy=sess.run([train_step,loss,accuracy],feed_dict={image_input:np.concatenate((im[start:],im[:BATCH_SIZE-im.shape[0]+start]),0),question_input:np.concatenate((qq[start:],qq[:BATCH_SIZE-qq.shape[0]+start]),0),answer_output:np.concatenate((aa[start:],aa[:BATCH_SIZE-aa.shape[0]+start]),0)})
			completed_epoch+=1
			start=BATCH_SIZE-im.shape[0]+start
			epoch_loss+=batch_loss
			epoch_accuracy+=batch_accuracy
			no+=1
			epoch_loss=epoch_loss/no
			epoch_accuracy=epoch_accuracy/no
			print('epoch%d:loss-%f,accuracy-%f'%(completed_epoch,epoch_loss,epoch_accuracy))
			epoch_loss=0
			epoch_accuracy=0
			no=0
end=time.time()
train_time=end-start
print('The time cost in train:%f' %train_time)
			
