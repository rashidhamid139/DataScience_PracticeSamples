#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
os.chdir(r"C:\Users\rashid\Desktop\datasets")


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.boxplot('dep_time', 'origin', rot=30, figsize=(5,6))


# In[121]:


cat_df = df.select_dtypes(include=['object']).copy()


# In[7]:


cat_df.info()


# In[8]:


cat_df.isnull().values.sum()


# In[9]:


cat_df.isnull().sum()


# In[10]:


cat_df = cat_df.fillna(cat_df['tailnum'].value_counts().index[0])


# In[11]:


cat_df


# In[12]:


cat_df['tailnum'].value_counts().index[0]


# In[13]:


cat_df.isnull().values.sum()


# In[14]:


cat_df['carrier'].value_counts().count()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
carrier_count = cat_df['carrier'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title("Frequency distribution of carriers")
plt.ylabel("No of Occurences", fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()



# In[ ]:





# In[86]:


labels  =cat_df['carrier'].astype('category').cat.categories.tolist()


# In[31]:


a


# In[87]:


replace_map_comp ={ 'carrier':{k:v for k,v in zip(labels, a)}}


# In[88]:


replace_map_comp


# In[81]:


cat_df_replace=cat_df.copy()


# In[89]:


cat_df_replace.replace(replace_map_comp, inplace=True)


# In[93]:


cat_df_replace['carrier'].dtypes


# In[94]:


cat_df_lc=cat_df.copy()


# In[110]:


cat_df_lc['carrier']=cat_df_lc['carrier'].astype('category')


# In[111]:


cat_df_lc.dtypes


# In[112]:


import time
get_ipython().run_line_magic('timeit', "cat_df.groupby(['origin','carrier']).count()")


# In[113]:


get_ipython().run_line_magic('timeit', "cat_df_lc.groupby(['origin', 'carrier']).count()")


# In[114]:


cat_df_lc['carrier']=cat_df_lc['carrier'].cat.codes


# In[120]:


cat_df


# In[122]:


cat_df_specific=cat_df.copy()
cat_df_specific.dtypes


# In[123]:


cat_df_specific['Us_code']=np.where(cat_df_specific['carrier'].str.contains('US'),1,0)


# In[130]:


cat_df_sklearn = cat_df.copy()
from sklearn.preprocessing import LabelEncoder
lb_make=LabelEncoder()
cat_df_sklearn['carrier_code'] = lb_make.fit_transform(cat_df['carrier'])
cat_df_sklearn.head()


# In[135]:


cat_df_onehot.head()


# In[138]:


cat_df_onehot_sklearn = cat_df.copy()


# In[149]:


cat_df_flights_onehot_sklearn = cat_df.copy()

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

print(lb_results_df.head())


# In[151]:


result_df = pd.concat([cat_df_flights_onehot_sklearn, lb_results_df], axis =1)


# In[153]:


result_df.head()


# In[155]:


dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})
dummy_df_age['start'], dummy_df_age['end'] = zip(*dummy_df_age['age'].map(lambda x: x.split('-')))

dummy_df_age.head()


# In[ ]:





# In[ ]:




