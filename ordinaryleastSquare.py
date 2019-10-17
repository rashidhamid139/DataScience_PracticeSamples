#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
from sklearn import datasets


# In[3]:


data = datasets.load_boston()


# In[4]:


data.keys()


# In[7]:


print(data.DESCR)


# In[8]:


import numpy as np
import pandas as pd

df = pd.DataFrame(data.data, columns=data.feature_names)


# In[11]:


target = pd.DataFrame(data.target, columns=['MEDV'])


# In[15]:


X= df['RM']


# In[16]:


y=target['MEDV']


# In[17]:


model = sm.OLS(y,X).fit()


# In[19]:


predictions = model.predict(X)


# In[20]:


model.summary()


# In[21]:


X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[22]:


X = df[["RM", "LSTAT"]]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[ ]:




