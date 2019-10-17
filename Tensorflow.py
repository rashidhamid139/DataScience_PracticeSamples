#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings


# In[2]:


import tensorflow as tf


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


tf.set_random_seed(1)
np.random.seed(1)


# In[41]:


x = np.linspace(-1,1,100)[:, np.newaxis]


# In[42]:


x


# In[43]:


noise=np.random.normal(0,0.1,size=x.shape)


# In[45]:


y=np.power(x,2)+noise


# In[48]:


plt.scatter(x,y)
plt.show()


# In[49]:


tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)


# In[50]:


l1 = tf.layers.dense(tf_x,10,tf.nn.relu)


# In[51]:


output = tf.layers.dense(l1, 1)


# In[52]:


loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train_op = optimizer.minimize(loss)


# In[53]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for step in range(100):
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step%5 == 0:
        plt.cla()
        plt.scatter(x,y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%.4f' % l, fontdict = {'size': 20, 'color':'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()


# In[55]:


n_data= np.ones((100,2))


# In[56]:


x0 = np.random.normal(2*n_data,1)


# In[57]:


x0


# In[58]:


y0 = np.zeros(100)                     # class0 y shape =(100,1)
x1 = np.random.normal(-2*n_data, 1)    # class1 x shape =(100,2)
y1 = np.ones(100)                      # class1 y shape =(100,1)
# vertical stacking
x = np.vstack((x0, x1))    # shape (200,2) + some noise
# horizontal stacking
y = np.hstack((y0, y1))    # shape (200, )

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()


# In[59]:


tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.int32, y.shape)


# In[60]:


l1 = tf.layers.dense(tf_x, 10, tf.nn.relu) # hidden layer
output = tf.layers.dense(l1, 2)           # output layer

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)    # cost function
# return (acc, update_op), and create 2 local variable
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)


# In[61]:


sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

plt.ion()
for step in range(100):
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
    if step%2 == 0:
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()


# In[62]:


LR = 0.01
BATCH_SIZE = 32


# In[63]:


x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot dataset
plt.scatter(x, y)
plt.show()


# In[ ]:


class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None,1])
        self.y = tf.placeholder(tf.float32, [None,1])
        l = tf.layers.dense(self.x, 20, tf.nn.relu)
        out = tf.layers.dense(l,1)
        self.loss = tf.losses.mean_squared_error(self.y, out)
        self.train = opt(LR, **kwargs).minimize(self.loss)
        net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]


# In[ ]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_his = [[],[],[]]          # recode loss

#training
for step in range(300):
    index = np.random.randint(0, x.shape[0], BATCH_SIZE)
    b_x = x[index]
    b_y = y[index]
    
    for net, l_his in zip(nets, losses_his):
        _, l = sess.run([net.train, net.loss], {net.x: b_x, net.y: b_y})
        l_his.append(l)    #loss recoder
        
#plot loss history
labels = ['SGD', 'Momentum','RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label = labels[i])
plt.legend(loc = 'best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()


# In[ ]:




