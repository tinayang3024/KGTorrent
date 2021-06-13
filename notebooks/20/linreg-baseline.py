#!/usr/bin/env python
# coding: utf-8

# # A simple linear baseline for the Walmart challenge
# This notebook shows how you load the data, prepare it for usage with Keras and then create a submission file. The model is a simple linear regression.

# In[1]:


import pandas as pd
import numpy as np


# ## Loading the data
# In Kaggle, data that can be accessed by a Kernel is saved under ``../inputs/``
# From there we can load it with pandas:

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# We are going to do some data preparation. It is easiest to do this for training and test set combined so we have to do all these steps only once. It is good to know where to split the set afterwards though!

# In[3]:


len(train) # Get number of training examples


# In[4]:


len(test) # Get number of test examples


# In[5]:


df = pd.concat([train,test],axis=0) # Join train and test


# In[6]:


df.head() # Get an overview of the data


# In[7]:


df.describe()


# There seem to be some missing values in the data. We have to make sure to deal with them before feeding anything into the network.

# In[8]:


df.isnull().sum()


# We will do a bit of very basic feature engineering here by creating a feature which indicates whether a certain markdown was active at all.

# In[9]:


df = df.assign(md1_present = df.MarkDown1.notnull())
df = df.assign(md2_present = df.MarkDown2.notnull())
df = df.assign(md3_present = df.MarkDown3.notnull())
df = df.assign(md4_present = df.MarkDown4.notnull())
df = df.assign(md5_present = df.MarkDown5.notnull())


# In[10]:


df.isnull().sum()


# We can probably safely fill all missing values with zero. For the markdowns this means that there was no markdown. For the weekly sales, the missing values are the ones we have to predict, so it does not really matter what we fill in there.

# In[11]:


df.fillna(0, inplace=True)


# In[12]:


df.dtypes


# Now we have to create some dummy variebles for categorical data.

# In[13]:


# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)


# In[14]:


# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])


# In[15]:


# Add dummies
df = pd.concat([df,type_dummies,store_dummies,dept_dummies],axis=1)


# In[16]:


# Remove originals
del df['Type']
del df['Store']
del df['Dept']


# In[17]:


del df['Date']


# In[18]:


df.dtypes


# Now we can split train test again.

# In[19]:


train = df.iloc[:282451]
test = df.iloc[282451:]


# In[29]:


test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test


# To get numpy arrays out of the pandas data frame, we can ask for a columns, or dataframes values

# In[21]:


y = train['Weekly_Sales'].values


# In[22]:


X = train.drop('Weekly_Sales',axis=1).values


# In[23]:


X.shape


# Now we create the baseline model

# In[24]:


from keras.layers import Dense, Activation
from keras.models import Sequential


# In[25]:


model = Sequential()
model.add(Dense(1,input_dim=145))
model.compile(optimizer='adam', loss='mae')


# In[26]:


model.fit(X,y,batch_size=2048,epochs=5)


# After we have created our model, we can predict things with it on the test set

# In[31]:


X_test = test.values


# In[32]:


y_pred = model.predict(X_test,batch_size=2048)


# To create the ids required for the submission we need the original test file one more time

# In[35]:


testfile = pd.read_csv('../input/test.csv')


# Now we create the submission. Once you run the kernel you can download the submission from its outputs and upload it to the Kaggle InClass competition page.

# In[38]:


submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':y_pred.flatten()})


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




