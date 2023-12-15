#!/usr/bin/env python
# coding: utf-8

# ## Week Three: Exploring Data In Multiple Ways

# ### Scikit-Learn

# In[1]:


from sklearn import datasets


# In[2]:


dir (datasets)


# datasets chosen are; load_digits load_iris load_wine because i thought they would be interesting 

# In[4]:


digits_data = datasets.load_digits()


# In[5]:


print (digits_data.DESCR)


# In[6]:


iris_data = datasets.load_iris()


# In[7]:


print (iris_data.DESCR)


# In[8]:


wine_data = datasets.load_wine()


# In[9]:


print (wine_data.DESCR)


# In[10]:


wine_data.feature_names


# In[11]:


wine_data.target_names


# In[12]:


from sklearn import datasets
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data["data"], columns = wine_data["feature_names"])


# In[13]:


wine_dataframe.head()


# In[14]:


wine_dataframe.describe()


# In[ ]:




