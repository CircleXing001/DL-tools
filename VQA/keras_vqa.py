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

import os
import gc
import time
from PIL import Image
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
KTF.set_session(tf.Session(config=config))

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

vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL_NUM)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())
image_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL_NUM))
encoded_image = vision_model(image_input)
question_input = Input(shape=(SENTENCE_LENGTH,), dtype='int32')
embedded_question = Embedding(input_dim=WORD_NUM, output_dim=256, input_length=SENTENCE_LENGTH)(question_input)
encoded_question = LSTM(256)(embedded_question)
merged = keras.layers.concatenate([encoded_question, encoded_image])
output = Dense(WORD_NUM, activation='softmax')(merged)
vqa_model = Model(inputs=[image_input, question_input], outputs=output)
vqa_model.compile(optimizer=OPTIM, loss='categorical_crossentropy',metrics=['accuracy'])
start=time.time()
vqa_model.fit([im,qq],aa,batch_size=BATCH_SIZE,epochs=EPOCH_NUM,verbose=1)
end=time.time()
train_time=end-start
print('The time cost in train:%f' %train_time)
