#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import copy
import os
get_ipython().run_line_magic('matplotlib', 'inline')
os.chdir(r"C:\Users\rashid\Desktop\Datasets")


# In[4]:


df_flights =pd.read_csv("https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv")


# In[5]:


df_flights.head()


# In[6]:


df_flights.info()


# In[7]:


df_flights.boxplot('dep_time','origin',rot = 30,figsize=(5,6))


# In[8]:


cat_df_flights = df_flights.select_dtypes(include=['object']).copy()


# In[9]:


cat_df_flights


# In[10]:


cat_df_flights.isnull().sum()


# In[11]:


cat_df_flights = cat_df_flights.fillna(cat_df_flights['tailnum'].value_counts().index[0])


# In[12]:


cat_df_flights.isnull().sum()


# In[13]:


cat_df_flights['tailnum'].value_counts().count()


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
carrier_count = cat_df_flights['origin'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Carriers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Tailnum', fontsize=12)
plt.show()


# In[18]:


labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
counts = cat_df_flights['carrier'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[36]:


cat_df_flights.head()


# #Methods to encode categorical features to numerical value
# 1.Replacing values
# 2. Encoding labels
# 3.One-Hot encoding
# 4. Binary Encoding
# 5. Backward Diffrence encoding
# 6. Miscelaneous features
# 

# In[33]:


cat_df_flights['carrier'].value_counts().count()


# In[34]:


cat_df_flights['tailnum'].value_counts().count()


# In[37]:


cat_df_flights['dest'].value_counts().count()


# In[41]:


cat_df_flights['origin'].value_counts()


# In[88]:


labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
replace_map_comp = {'carrier':{k: v for k,v in zip(labels, list(range(1,len(labels)+1)))}}


# In[89]:


cat_df_flights_replace = cat_df_flights.copy()


# In[90]:


cat_df_flights_replace.head()


# In[91]:


cat_df_flights_replace.replace(replace_map_comp, inplace=True)


# In[93]:


cat_df_flights_replace.head(10)


# In[97]:


cat_df_flights_replace['dest'].dtypes


# In[98]:


cat_df_flights_lc = cat_df_flights.copy()
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].astype('category')
cat_df_flights_lc['origin'] = cat_df_flights_lc['origin'].astype('category')                                                              


# In[99]:


cat_df_flights_lc.dtypes


# In[100]:


import time
get_ipython().run_line_magic('timeit', "cat_df_flights.groupby(['origin', 'carrier']).count()")


# In[101]:



get_ipython().run_line_magic('timeit', "cat_df_flights_lc.groupby(['origin', 'carrier']).count()")


# #Label Encoding
# 

# In[111]:


cat_df_flights_lc['carrier']=cat_df_flights_lc['carrier'].cat.codes


# In[117]:


cat_df_flights_specific=cat_df_flights.copy()


# In[120]:


cat_df_flights_specific['US_code']=np.where(cat_df_flights_specific['carrier'].str.contains('US'), 1,0)
cat_df_flights_specific.head()


# In[122]:


cat_df_flights_sklearn= cat_df_flights.copy()


# In[123]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()


# In[125]:


cat_df_flights_sklearn['carrier_code'] = lb_make.fit_transform(cat_df_flights['carrier'])


# In[127]:


cat_df_flights_sklearn.head()


# In[128]:


cat_df_flights_onehot= cat_df_flights.copy()


# In[131]:


cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix=['carrier'])


# In[132]:


cat_df_flights_onehot.head()


# In[133]:


cat_df_flights_onehot_sklearn=cat_df_flights.copy()


# In[135]:


cat_df_flights_onehot_sklearn.head()


# In[138]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)


# In[139]:


lb_results_df.head()


# In[141]:


result_df= pd.concat([cat_df_flights_onehot_sklearn, lb_results_df], axis=1)


# In[144]:


result_df.head()


# Binary Encoding

# In[145]:


cat_df_flights_ce = cat_df_flights.copy()


# In[148]:


import category_encoders as ce


# In[149]:


ncoder = ce.BackwardDifferenceEncoder(cols=['carrier'])
df_bd = encoder.fit_transform(cat_df_flights_ce)

df_bd.head()


# In[150]:


dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})


# In[151]:


dummy_df_age


# In[153]:


dummy_df_age['start'], dummy_df_age['end']=zip(*dummy_df_age['age'].map(lambda x:x.split('-')))


# In[154]:


dummy_df_age


# In[ ]:





# In[164]:


class MovingAverage():
    def __init__(self,symbol, bars, short_window, long_window):
        self.symbol=symbol
        self.bars=bars
        self.short_window=short_window
        self.long_window=long_window
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        signals['short_mavg'] =self.bars['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.bars['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)   

        signals['positions'] = signals['signal'].diff()   

        return sign


# In[191]:


def C(c):
    return c*9/5 +32
print(C(100))
print(C(0))


# In[192]:


apple=mango


# In[168]:


microsoft = MovingAverage('msft', msft, 40, 100)
print(microsoft.generate_signals())


# In[193]:


liste=[3,4,5,20,5,25,1,3]


# # 

# In[194]:


liste.pop(1)


# In[195]:


liste


# In[ ]:




