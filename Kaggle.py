#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
os.chdir(r"C:\Users\rashid\Desktop\datasets")


# In[3]:


df=pd.read_csv("train.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


cat_df = df.select_dtypes(include=['object']).copy()


# In[16]:


cat_df.head()


# In[15]:


cat_df.isnull().values.sum()


# In[17]:


cat_df.columns


# In[26]:


cat_df['LandSlope'].value_counts().count()


# In[25]:


cat_df['Neighborhood'].value_counts().count()


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
carrier_count = cat_df['HouseStyle'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title("Frequency distribution of carriers")
plt.ylabel("No of Occurences", fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()



# In[35]:





# In[45]:


counts = cat_df['HouseStyle'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




