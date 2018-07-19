import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as da
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
   
x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels  
x_train = x_train.reshape(-1, 1,28, 28).astype('float32')  
x_test = x_test.reshape(-1,1,28, 28).astype('float32')
y_train=y_train.astype('float32')
y_test=y_test.astype('float32')
x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)
x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
train_data=da.TensorDataset(x_train,y_train)
test_data=da.TensorDataset(x_test,y_test)
train_loader = da.DataLoader(train_data,batch_size=50,shuffle=True)
test_loader = da.DataLoader(test_data,batch_size=50)
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1=nn.Conv2d(1,32,(5,5),padding=(2,2))
		self.pool1=nn.MaxPool2d((2,2),stride=(2,2))
		self.conv2=nn.Conv2d(32,64,(5,5),padding=(2,2))
		self.pool2=nn.MaxPool2d((2,2),stride=(2,2))
		self.fc1=nn.Linear(7*7*64,1024)
		self.drop1=nn.Dropout(p=0.5)
		self.fc2=nn.Linear(1024,10)
	def forward(self,x):
		x=self.conv1(x)
		x=F.relu(x)
		x=self.pool1(x)
		x=self.conv2(x)
		x=F.relu(x)
		x=self.pool2(x)
		x=x.view(-1,7*7*64)
		x=self.fc1(x)
		x=F.relu(x)
		x=self.drop1(x)
		x=self.fc2(x)
		x=F.softmax(x)
		return x
model=Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
def train(epoch_num):
	model.train()
	for epoch in range(epoch_num):
		for batch_idx, (data, target) in enumerate(train_loader):
			optimizer.zero_grad()
			output=model(data)
			loss=-torch.sum(target*torch.log(output+1e-10))
			loss.backward()
			optimizer.step()
			output1=output.detach().numpy()
			target1=target.numpy()
			accuracy=np.equal(np.argmax(output1,1), np.argmax(target1,1))
			accuracy=accuracy.astype('float')
			accuracy=np.mean(accuracy)
			print(batch_idx*50,float(loss),accuracy)
train_start=time.time()
train(10)
train_end=time.time()
train_time=train_end-train_start
print('The time cost in train:%f'%train_time)
