#!/usr/bin/env python
# coding: utf-8

# SVM Algorithm:
# ->It is a Supervised Macine Learning algorithm
# -> SVM offers very high accuracy as compared to other classifiers such as     Logistic Regression and Decision trees
# 
# ->It is used in variety of application such as face detection , intrusion    detection, classification of emails, news and articles and webpages, classification of genes, and handwritten recognition
# 
# ->SVM is usally considered to be a classification alogo, but it can be employed in both types of classification and regression problems.

# How SVM work?
# -> The main objective is to segregate the given dataset in the best possible way.
# -> The objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset

# Kernel Trick:
# -> Linear Kernel: 
#     K(x,xi) = sum(x*xi)
#     
# -> Polynomial kernel:
#     K(x,xi) = 1+ sum(x*xi)^d
#     where d is the degree of polynomial
#     
# -> Radial Basis Kernel Function:
#     k(x,xi) = exp(-gamma * sum((x-xi^2))
#     gamma range is 0 to 1
#     a higher value of Gamma results in overfitting
#     
#     

# In[195]:


from sklearn import datasets
cancer = datasets.load_breast_cancer()


# In[205]:


cancer.keys()


# In[208]:


cancer.data[0:1].shape


# In[211]:


cancer.target[0:5]


# In[212]:


from sklearn.model_selection import train_test_split


# In[213]:


X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)


# Import SVM module and create support vector classifier object by passing argument kernel as the linear kernel in SVC() function.

# In[215]:


from sklearn import svm

clf = svm.SVC(kernel='linear')


# In[216]:


clf.fit(X_train, Y_train)


# In[217]:


y_pred=clf.predict(X_test)


# In[219]:


y_pred


# In[221]:


from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))


# In[222]:


print("Precision:", metrics.precision_score(Y_test, y_pred))


# In[223]:


print("Recall:", metrics.recall_score(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




