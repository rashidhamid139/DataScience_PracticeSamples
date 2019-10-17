#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


tips = sns.load_dataset('tips')


# In[9]:


sns.barplot(x='day', y='tip', data = tips, palette='winter_r')


# In[10]:


sns.barplot(x='day', y='total_bill', data = tips, hue='sex')


# In[12]:


sns.barplot(x='day', y='tip', data = tips)


# In[14]:


sns.barplot(x='total_bill', y='day', data = tips, palette='spring')


# In[18]:


sns.barplot(x='day', y='tip', data = tips, palette='spring', order=['Sat', 'Fri', 'Sun', 'Thur'])


# In[21]:


from numpy import median
sns.barplot(x='day',y='total_bill', data=tips, estimator=median, palette='spring')


# In[24]:


sns.barplot(x='smoker', y='tip', data=tips, ci=99)


# In[29]:


sns.barplot(x='smoker', y='tip', data=tips, ci=34, palette='winter_r', estimator=median)


# In[32]:


sns.barplot(x='day', y='total_bill', data=tips, palette='spring', capsize=0.9)


# In[59]:


sns.barplot(x='size', y='tip', data=tips, color='blue',saturation=0.9 ,capsize=0.9)


# In[ ]:





# In[60]:


tips= sns.load_dataset('tips')


# In[64]:


sns.boxplot(x=tips['total_bill'])


# In[74]:


sns.boxplot(x='day', y='total_bill', data=tips, hue='time', palette='coolwarm')


# In[75]:


#HeatMap


# In[83]:


normal = np.random.rand(10,12)


# In[89]:


sns.heatmap(normal, vmin=0, vmax=2)


# In[94]:


flights = sns.load_dataset('flights')
f=flights.pivot('month','year', 'passengers')


# In[95]:


sns.heatmap(f)


# In[ ]:




