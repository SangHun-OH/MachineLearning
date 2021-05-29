#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        output = self.fc(r_out)

        return output, hidden


# In[3]:


#data structure
def sin_data(x, T=100):
    return np.sin(2.0*np.pi*x/T)

def toy_problem(T=100, amp = 0.05):
    x = np.arange(0, 2*T + 1)
    return sin_data(x, T)

T = 100
f = toy_problem(T)

data = []
target = []

for i in range(0, T-25+1):
    data.append(f[i:i+25])
    target.append(f[i+25])

data = torch.Tensor(data)
target = torch.Tensor(target)


# In[4]:


# train, pred list
train = data[0]
pred = data[0]
train = (train.unsqueeze(0)).unsqueeze(0)
data = data.unsqueeze(0)


# In[5]:


# 하이퍼 파라미터
input_size = 25
output_size = 1
hidden_size = 32
n_layers = 1


# In[6]:


# instance
rnn = RNN(input_size, output_size, hidden_size, n_layers)


# In[7]:


#손실함수와 최적화
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


# In[8]:


# predict
predict = np.zeros(300)


# In[9]:


#training
hidden = None

for i in range(600):
    optimizer.zero_grad()
    model, hidden = rnn(data, hidden)
    hidden = hidden.data
    loss = loss_function(model.squeeze(), target)
    loss.backward()
    optimizer.step()

hidden = None

with torch.no_grad():
    for i in range(25,300):
        pred, hidden = rnn(train, hidden)
        hidden = hidden.data
        train = torch.cat([train[:,:,1:25],pred],dim=2)
        predict[i]=pred.squeeze()


# In[10]:


plot = data[0][0].view([-1])
default = torch.cat([plot, target[0:75]], dim=0)
default_2 = torch.cat([default, default], dim=0)
default = torch.cat([default, default_2], dim=0)


# In[11]:


plt.figure(figsize=(8,8))
plt.plot(default, label='Default')
plt.plot(predict, label='Predict')
plt.legend()
plt.show()


