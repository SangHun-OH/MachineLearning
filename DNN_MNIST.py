#!/usr/bin/env python
# coding: utf-8

# # 10조
# 오상훈 / 2016125039 / 팀원: 김태훈, 백윤성, 이해인

# In[1]:


import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


# In[2]:


# 모델 구성하기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)
        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)

        return x5
 


# In[3]:


#xavier를 적용하기 위한 함수
def xavier_init(model):
    torch.nn.init.xavier_uniform_(model.fc1.weight)
    torch.nn.init.xavier_uniform_(model.fc2.weight)
    torch.nn.init.xavier_uniform_(model.fc3.weight)


# In[4]:


# he(kaim)을 적용하기 위한 함수
def kaiming_init(model):
    torch.nn.init.kaiming_uniform_(model.fc1.weight)
    torch.nn.init.kaiming_uniform_(model.fc2.weight)
    torch.nn.init.kaiming_uniform_(model.fc3.weight)


# In[5]:


download_root = 'MNIST_data/'


# In[6]:


dataset1 = datasets.MNIST(root=download_root,
                         train=True,
                         transform = transforms.ToTensor(),
                         download=True)


# In[7]:


dataset2 = datasets.MNIST(root=download_root,
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


# In[8]:


# DataLoader를 이용해서 batch size 정하기(batch size는 100으로 지정)
batch_s = 100
dataset1_loader = DataLoader(dataset1, batch_size=batch_s, shuffle=True)
dataset2_loader = DataLoader(dataset2, batch_size=batch_s, shuffle=True)


# In[9]:


# 각 케이스에 대한 model 정보, loss, accuracy 정보를 저장하기 위한 딕셔너리 생성
model_dict = {}
loss_dict = {}
accuracy_dict = {}

# 총 12가지의 케이스를 정의(optimizer만 사용한 것, optimizer+초기화)
case_list = ['SGD','Adam','AdaGrad','RMSprop','SGDxavier',              'Adamxavier','AdaGradxavier','RMSpropxavier',             'SGDkaim','Adamkaim','AdaGradkaim','RMSpropkaim']

# optimizer이름을 key로 정함
for key in case_list:
    model_dict[key] = Net()
    loss_dict[key]=[]
    accuracy_dict[key]=[]

    # xavier와 he의 경우 위에서 선언한 함수를 호출하여 초기화 적용
    if 'xavier' in key:
        xavier_init(model_dict[key])
        model_dict[key] = nn.Sequential(model_dict[key].fc1, model_dict[key].relu,                                        model_dict[key].fc2, model_dict[key].relu,                                       model_dict[key].fc3)
    elif 'kaim' in key:
        kaiming_init(model_dict[key])
        model_dict[key] = nn.Sequential(model_dict[key].fc1, model_dict[key].relu,                                        model_dict[key].fc2, model_dict[key].relu,                                       model_dict[key].fc3)


# In[10]:


# 위에서 정의한 12가지 모델에 대해서 각각 optimizer 설정하기
learning_rate = 0.01
optimizer_dict = {}

optimizer_dict['SGD'] = optim.SGD(model_dict['SGD'].parameters(), lr=learning_rate)
optimizer_dict['SGDxavier'] = optim.SGD(model_dict['SGDxavier'].parameters(), lr=learning_rate)
optimizer_dict['SGDkaim'] = optim.SGD(model_dict['SGDkaim'].parameters(), lr=learning_rate)

optimizer_dict['Adam'] = optim.Adam(model_dict['Adam'].parameters(), lr=learning_rate)
optimizer_dict['Adamxavier'] = optim.Adam(model_dict['Adamxavier'].parameters(), lr=learning_rate)
optimizer_dict['Adamkaim'] = optim.Adam(model_dict['Adamkaim'].parameters(), lr=learning_rate)

optimizer_dict['AdaGrad'] = optim.Adagrad(model_dict['AdaGrad'].parameters(), lr=learning_rate)
optimizer_dict['AdaGradxavier'] = optim.Adagrad(model_dict['AdaGradxavier'].parameters(), lr=learning_rate)
optimizer_dict['AdaGradkaim'] = optim.Adagrad(model_dict['AdaGradkaim'].parameters(), lr=learning_rate)

optimizer_dict['RMSprop'] = optim.RMSprop(model_dict['RMSprop'].parameters(), lr=learning_rate)
optimizer_dict['RMSpropxavier'] = optim.Adagrad(model_dict['RMSpropxavier'].parameters(), lr=learning_rate)
optimizer_dict['RMSpropkaim'] = optim.Adagrad(model_dict['RMSpropkaim'].parameters(), lr=learning_rate)


# In[11]:


# loss function, total_batch size, epoch 정의
loss_function = nn.CrossEntropyLoss()
total_batch = len(dataset1_loader)
epochs = np.arange(1,16)
print(len(dataset1_loader)) #60000개의 data를 batch_size를 100으로 했기 때문에 600이 나오는 것. 


# In[12]:


# 총 12가지 케이스의 loss와 accuracy를 한 번에 담아주려고 한다.
# optimizer_name에는 optimizer_dict로 들어간 키 값이, optimizer에는 optimizer를 적용한 내용이 들어간다.
for optimizer_name, optimizer in optimizer_dict.items(): #총 15회 학습
    model_dict[optimizer_name].zero_grad()
    #print("optimizer : ",optimizer_name)
    for epoch in epochs:
        cost=0
        for images, labels in dataset1_loader:
            images = images.reshape(100,784)

            optimizer.zero_grad() # 변화도 매개변수 0
            
            #초기화를 사용하지 않는 모델들에 대한 forward
            if(optimizer_name == 'SGD' or optimizer_name=='Adam'                or optimizer_name=='AdaGrad' or optimizer_name == 'RMSprop'):
                pred = model_dict[optimizer_name].forward(images)
            elif('xavier' in optimizer_name): 
                #xavier 초기화 방법 사용
                pred = model_dict[optimizer_name](images)
            elif('kaim' in optimizer_name):
                #he(kaim) 초기화 방법 사용
                pred = model_dict[optimizer_name](images)

            # loss값 구하기
            loss = loss_function(pred, labels)
            
            #backward(Back propagation)
            loss.backward()

            #Update(weight update 진행)
            optimizer.step()

            # 600번을 반복한 loss의 결과가 다 더해짐
            cost += loss

        # 제대로 학습이 되었는지 테스트 해보는 과정
        with torch.no_grad(): #미분하지 않겠다는 것
            total = 0
            correct=0
            for images, labels in dataset2_loader:
                images = images.reshape(100,784)

                outputs = model_dict[optimizer_name](images)
                _,predict = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predict==labels).sum() # 예측한 값과 일치한 값의 합

        avg_cost = cost / total_batch
        accuracy = 100*correct/total
        
        loss_dict[optimizer_name].append(avg_cost.detach().numpy())
        accuracy_dict[optimizer_name].append(accuracy)
        
        #print("epoch : {} | loss : {:.6f}" .format(epoch, avg_cost))
        #print("Accuracy : {:.2f}".format(100*correct/total))
        #print("------")
print("finished")


# # 과제 1. 지난주에 만들었던 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# - optimizer를 SGD로 사용했을 때 epoch에 따른 loss와 accuracy 변화를 나타낸 것이다. 
# - 학습이 진행될수록 loss 값을 줄어들고, accuracy는 증가한다.

# In[13]:


# 과제 1. 지난주에 만들었던 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# optimizer를 SGD로 사용했을 때 epoch에 따른 loss와 accuracy 변화
# 학습이 진행될수록 loss 값을 줄어들고, accuracy는 증가한다.
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs,loss_dict['SGD'])
plt.subplot(1,2,2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, accuracy_dict['SGD'])
plt.suptitle("Homework 1 SGD loss vs epoch & accuracy vs epoch", fontsize=20)
plt.show()


# # 과제 2. 위 주제 중 1개를 반영하여 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# - optimizer로 adam, adagrd, RMSprop를 사용했을 때 epoch에 따른 loss와 accuracy 변화를 나타내었다.
# - adagrad를 적용했을 때 가장 안정적으로 loss가 줄어들고 정확도가 증가하였다.
# 

# In[14]:


# 과제 2. 위 주제 중 1개를 반영하여 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# optimizer로 adam, adagrd, RMSprop를 사용했을 때 epoch에 따른 loss와 accuracy 변화
# adagrad를 적용했을 때 가장 안정적으로 loss가 줄어들고 정확도가 증가
markers =  {"SGD": "o","RMSprop": "x", "AdaGrad": "s", "Adam": "D"}

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for key in optimizer_dict.keys():
    if key=='RMSprop' or key=='AdaGrad' or key=='Adam':
        plt.plot(epochs, loss_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
for key in optimizer_dict.keys():
    if key=='RMSprop' or key=='AdaGrad' or key=='Adam':
        plt.plot(epochs, accuracy_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.suptitle("Homework 2 SGD loss vs epoch & accuracy vs epoch", fontsize=20)
plt.show()


# # 과제 3. 다른 1개를 반영하여 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# - optimizer SGD에 xavier 방식과 he 방식을 각각 적용해봤을 때 epoch에 따른 loss와 accuracy 출력하였다.
# - 초기화 방식으로 xavier보다 he(kaim)을 썼을 때 더 좋은 성능을 보였다.

# In[15]:


# 과제 3. 위 주제 중 다른 1개를 반영하여 인공신경망의 Loss vs 학습량, 정확도 vs 학습량
# optimizer SGD에 xavier 방식과 he 방식을 각각 적용해봤을 때 epoch에 따른 loss와 accuracy 출력
# 초기화 방식으로 he(kaim)을 썼을 때 더 좋은 성능을 보였다.
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs,loss_dict['SGDxavier'],'b',label='SGD & Xavier')
plt.plot(epochs,loss_dict['SGDkaim'],'r',label='SGD & kaiming')
plt.legend()

plt.subplot(1,2,2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, accuracy_dict['SGDxavier'],'b',label='SGD & Xavier')
plt.plot(epochs, accuracy_dict['SGDkaim'],'r',label='SGD & kaiming')
plt.legend()

plt.suptitle("Homework 3 SGD initialization loss vs epoch & accuracy vs epoch", fontsize=20)
plt.show()


# # 과제 4. 위 주제 중 2개를 반영하여 인공 신경망의 Loss vs 학습량, 정확도 vs 학습량
# - 위쪽 두 개의 그래프는 optimizer 중 adam, adagrad, RMSprop에 xavier를 적용한 뒤 epoch에 따른 loss와 accuracy의 변화를 나타낸 것이다. 
# - 아래쪽 두 개의 그래프는 adam, adagrad, RMSprop 각각에 he 방식인 kaim을 적용한 뒤에 epoch에 따른 loss와 accuracy의 변화를 나타낸 것이다.
# - 전반적으로 xavier보다는 he 방식을 사용했을 때 더 좋은 성능을 보였다. 또한 학습에 따라 약간의 차이는 있었지만 adagrad와 RMSprop에 he를 적용하였을 때 평균적으로 adagrad보다는 RMSprop가 loss가 더 낮고, 정확도가 더 높았다.

# In[16]:


# 과제 4. 위 주제 중 2개를 반영하여 인공 신경망의 Loss vs 학습량, 정확도 vs 학습량
markers =  {"SGDxavier": "o","RMSpropxavier": "x", "AdaGradxavier": "s", "Adamxavier": "D"}

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
for key in optimizer_dict.keys():
    if key=='RMSpropxavier' or key=='AdaGradxavier' or key=='Adamxavier' :
        plt.plot(epochs, loss_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,2)
for key in optimizer_dict.keys():
    if key=='RMSpropxavier' or key=='AdaGradxavier' or key=='Adamxavier':
        plt.plot(epochs, accuracy_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

markers =  {"SGDkaim": "o","RMSpropkaim": "x", "AdaGradkaim": "s", "Adamkaim": "D"}
plt.subplot(2,2,3)
for key in optimizer_dict.keys():
    if key=='RMSpropkaim' or key=='AdaGradkaim' or key=='Adamkaim' :
        plt.plot(epochs, loss_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,4)
for key in optimizer_dict.keys():
    if key=='RMSpropkaim' or key=='AdaGradkaim' or key=='Adamkaim':
        plt.plot(epochs, accuracy_dict[key], marker=markers[key], markevery=100, label=key)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.suptitle("Homework 4 loss vs epoch & accuracy vs epoch", fontsize=20)
plt.show()


# In[ ]:




