#!/usr/bin/env python
# coding: utf-8

# # The Walmart challenge: Modelling weekly sales
# In this notebook, we use data from Walmart to forecast their weekly sales. 

# ## Summary of results and approach
# 
# Work in Progress:
# 
# At writing, our internal competition at Bletchley has ended. Interestingly, the winning group had a different approach then would be expected from an AI/Machine Learning bootcamp. Their forecasts were based simply on a median of the weekly sales grouped by the Type of Store, Store & Department number, Month and Holiday dummy. 
# 
# Therefore, in my next approach, the goal will be to improve their results with the help of Neural Networks. In fact, the median will be computed similarly to how the winning group did, and a new variable, the difference to the median, will be computed. This difference will be the new dependent variable and will be estimated based on new holiday dummies, markdowns and info on lagged sales data if available.
# 
# **Unfortunately, it appears that so far the models do not find any possible improvements over the median sales forecasts with the available explanatory variables.
# **

# ## Understanding the problem and defining a success metric
# 
# The problem is quite straightforward. Data from Walmart stores accross the US is given, and it is up to us to forecast their weekly sales. The data is already split into a training and a test set, and we want to fit a model to the training data that is able to forecast those weeks sales as accurately as possible. In fact, our metric of interest will be the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error). 
# 
# The metric is not very complicated. The further away from the actual outcome our forecast is, the harder it will be punished. Optimally, we exactly predict the weekly sales. This of course is highly unlikely, but we must try to get as close as possible. The base case of our model will be a simple linear regression baseline, which gave a MSE of 
# 
# 

# ## Load and explore data
# Before we do anything, lets import some packages.

# In[1]:


import pandas as pd
import numpy as np
from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


# Now, load the train and test data.

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In order to efficiently modify our data, we merge the two datasets for now. We also keep track of the length of our training set so we know how to split it later.

# In[3]:


t_len = len(train) # Get number of training examples
df = pd.concat([train,test],axis=0) # Join train and test
df.head() # Get an overview of the data


# Let's get a clearer image of what our data actually looks like with the describe function. This will give use summary statistics of our numerical variables.

# In[4]:


df.describe()


# Since we are in the Netherlands, and we don't understand Fahrenheit, let's do a quick change there.

# In[5]:


df['Temperature'] = (df['Temperature'] - 32) * 5/9


# Although there is not a large variety of variables, we can definitely work with this. In the next section, we will clean the data set, engineer some new features and add dummy variables. For now, let's try to find any obvious relations between our variables to get a feeling for the data. We begin with a correlation matrix.
# 

# In[6]:


# Code from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Most of what we see in the correlation table is of little surprise. Discounts are correlated and higher unemployment means lower Consumer Price Index. More interestingly, it appears that higher department numbers have higher sales. Maybe because they are newer? Also, larger stores generate more sales, discounts generally generate higher sales values and larger unemployment result in a bit fewer sales. Unfortunately, there appears to be little relationship between holidays, temperatures or fuelprices with our weekly sales.
# 
# Next up, let's plot some of these relationships to get a clearer image.

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
def scatterplots(feature, label):
    x = feature
    y = df['Weekly_Sales']
    plt.scatter(x, y)
    plt.ylabel('sales')
    plt.xlabel(label)
    plt.show()

headers = list(df)
labels = headers
scatterplots(df['Fuel_Price'], 'Fuel_Price')
scatterplots(df['Size'], 'Size')
scatterplots(df['Temperature'], 'Temperature')
scatterplots(df['Unemployment'], 'Unemployment')
scatterplots(df['IsHoliday'], 'IsHoliday')
scatterplots(df['Type'], 'Type')


# From this plot, we notice that type C stores have fewer sales in general and holidays clearly show more sales.Although no further relationships appear evident from this analysis, there appears to be some outliers in our data. Let's take a bit of a closer look at these.

# In[8]:


df.loc[df['Weekly_Sales'] >300000]


# It appears to be quite obvious. The end of November sees a lot of exceptionally large sales. This special day, better known as Black friday, causes sales to be on fire, and undoubtedly a dummy variable should be created for this day. Also, Christmas, appears here and there. Since it is not considered holiday, we will also make a dummy for this day. Let's see if we should consider some other special days as well.

# In[9]:


df.loc[df['Weekly_Sales'] >240000,"Date"].value_counts()


# Except for a handful spurious other dates, it appears that the two days before Christmas and Black Friday will do the job.

# 
# 
# ## Scrub the data and engineer features
# 
# ### Missing values
# 
# We will start with filling in any blank values. There seem to be some missing values in the data. We have to make sure to deal with them before feeding anything into the network.

# In[10]:


df.isnull().sum()


# We will do a bit of very basic feature engineering here by creating a feature which indicates whether a certain markdown was active at all.

# In[11]:


df = df.assign(md1_present = df.MarkDown1.notnull())
df = df.assign(md2_present = df.MarkDown2.notnull())
df = df.assign(md3_present = df.MarkDown3.notnull())
df = df.assign(md4_present = df.MarkDown4.notnull())
df = df.assign(md5_present = df.MarkDown5.notnull())


# We can probably safely fill all missing values with zero. For the markdowns this means that there was no markdown. For the weekly sales, the missing values are the ones we have to predict, so it does not really matter what we fill in there.

# In[12]:


df.fillna(0, inplace=True)


# ### Dummy variables: Categorical Data
# 
# Now we have to create some dummy variebles for categorical data.

# In[13]:


# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
df['IsHoliday'] = 'IsHoliday_' + df['IsHoliday'].map(str)


# In[14]:


# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
holiday_dummies = pd.get_dummies(df['IsHoliday'])


# ### Dummy variables: Dates
# 
# From our earlier analysis, it has turned out that the date may be our best friend. As a general rule, it is a good start to already distinguish between different months in our model. This will create 12 dummy variables; one for each month.

# In[15]:


df['DateType'] = [datetime.strptime(date, '%Y-%m-%d').date() for date in df['Date'].astype(str).values.tolist()]
df['Month'] = [date.month for date in df['DateType']]
df['Month'] = 'Month_' + df['Month'].map(str)
Month_dummies = pd.get_dummies(df['Month'] )


# Next, let's look at 'special dates'. One variable for Christmas, one for black friday. We have to manually look up the dates of black friday if we want to extrapolate our data to other years, but for now we know: 26 - 11 - 2010 and 25 - 11 - 2011.

# In[16]:


df['Black_Friday'] = np.where((df['DateType']==datetime(2010, 11, 26).date()) | (df['DateType']==datetime(2011, 11, 25).date()), 'yes', 'no')
df['Pre_christmas'] = np.where((df['DateType']==datetime(2010, 12, 23).date()) | (df['DateType']==datetime(2010, 12, 24).date()) | (df['DateType']==datetime(2011, 12, 23).date()) | (df['DateType']==datetime(2011, 12, 24).date()), 'yes', 'no')
df['Black_Friday'] = 'Black_Friday_' + df['Black_Friday'].map(str)
df['Pre_christmas'] = 'Pre_christmas_' + df['Pre_christmas'].map(str)
Black_Friday_dummies = pd.get_dummies(df['Black_Friday'] )
Pre_christmas_dummies = pd.get_dummies(df['Pre_christmas'] )


# In[17]:


# Add dummies
# We will actually skip some of these
#df = pd.concat([df,type_dummies,store_dummies,dept_dummies,holiday_dummies,Pre_christmas_dummies,Black_Friday_dummies,Month_dummies],axis=1)

df = pd.concat([df,holiday_dummies,Pre_christmas_dummies,Black_Friday_dummies],axis=1)


# > ### Store median
# 
# We will take the store median in the available data as one of its properties

# In[18]:


# Get dataframe with averages per store and department
medians = pd.DataFrame({'Median Sales' :df.iloc[:282451].groupby(by=['Type','Dept','Store','Month'])['Weekly_Sales'].median()}).reset_index()


# In[19]:



# Merge by type, store, department and month
df = df.merge(medians, how = 'outer', on = ['Type','Dept','Store','Month'])


# In[20]:


# Fill NA
df['Median Sales'].fillna(df['Median Sales'].iloc[:282451].median(), inplace=True) 

# Create a key for easy access

df['Key'] = df['Type'].map(str)+df['Dept'].map(str)+df['Store'].map(str)+df['Date'].map(str)


# In[21]:


df.head()


# ### Lagged Variables
# 
# We will take a lagged variable of our store's previous weeks sales. To do so, we will first add a column with a one week lagged date, sort the data, and then match the lagged sales with the initial dataframe using the department and store number.
# 
# We begin by adding a column with a one week lag.

# In[22]:


# Attach variable of last weeks time
df['DateLagged'] = df['DateType']- timedelta(days=7)
df.head()


# Next, we create a sorted dataframe.

# In[23]:


# Make a sorted dataframe. This will allow us to find lagged variables much faster!
sorted_df = df.sort_values(['Store', 'Dept','DateType'], ascending=[1, 1,1])
sorted_df = sorted_df.reset_index(drop=True) # Reinitialize the row indices for the loop to work


# Loop over its rows and check at each step if the previous week's sales are available. If not, fill with store and department average, which we retrieved before.

# In[24]:


sorted_df['LaggedSales'] = np.nan # Initialize column
last=df.loc[0] # intialize last row for first iteration. Doesn't really matter what it is
row_len = sorted_df.shape[0]
for index, row in sorted_df.iterrows():
    lag_date = row["DateLagged"]
    # Check if it matches by comparing last weeks value to the compared date 
    # And if weekly sales aren't 0
    if((last['DateType']== lag_date) & (last['Weekly_Sales']>0)): 
        sorted_df.set_value(index, 'LaggedSales',last['Weekly_Sales'])
    else:
        sorted_df.set_value(index, 'LaggedSales',df['Median Sales'].loc[index]) # Fill with median

    last = row #Remember last row for speed
    if(index%int(row_len/10)==0): #See progress by printing every 10% interval
        print(str(int(index*100/row_len))+'% loaded')


# In[25]:


sorted_df.head()


# Now, merge this new info with our existing dataset.

# In[26]:


# Merge by store and department
df = df.merge(sorted_df[['Dept', 'Store','DateType','LaggedSales']], how = 'inner', on = ['Dept', 'Store','DateType'])


# ### Remove redundant items
# 
# We will take the store average in the available data as one of its properties

# In[27]:


# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['IsHoliday']
del df['DateType']
del df['Date']
del df['Month']
del df['Pre_christmas'] 
del df['Black_Friday']
del df['DateLagged']
del df['Key']


# ### Scale Variables
# 
# To make the job of our models easier in the next phase, we normalize our continous data. This is also called feature scaling.

# In[28]:


df['Unemployment'] = (df['Unemployment'] - df['Unemployment'].mean())/(df['Unemployment'].std())
df['Temperature'] = (df['Temperature'] - df['Temperature'].mean())/(df['Temperature'].std())
df['Fuel_Price'] = (df['Fuel_Price'] - df['Fuel_Price'].mean())/(df['Fuel_Price'].std())
df['CPI'] = (df['CPI'] - df['CPI'].mean())/(df['CPI'].std())
df['MarkDown1'] = (df['MarkDown1'] - df['MarkDown1'].mean())/(df['MarkDown1'].std())
df['MarkDown2'] = (df['MarkDown2'] - df['MarkDown2'].mean())/(df['MarkDown2'].std())
df['MarkDown3'] = (df['MarkDown3'] - df['MarkDown3'].mean())/(df['MarkDown3'].std())
df['MarkDown4'] = (df['MarkDown4'] - df['MarkDown4'].mean())/(df['MarkDown4'].std())
df['MarkDown5'] = (df['MarkDown5'] - df['MarkDown5'].mean())/(df['MarkDown5'].std())
df['LaggedSales']= (df['LaggedSales'] - df['LaggedSales'].mean())/(df['LaggedSales'].std())


# Now, let's change the variable to be forecasted to the difference from the median. Afterward, we can drop the weekly sales.

# In[29]:


df['Difference'] = df['Median Sales'] - df['Weekly_Sales']
del df['Weekly_Sales'] 


# Let's have a look at our data set before running our actual models.

# In[30]:


df.head()


# In[31]:


# Code from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Select variables to include in model
# 
# In this section, we can change the variables we ultimately want to include in our model training. 

# In[32]:


selector = [
    'CPI',
    'Fuel_Price',
    #'MarkDown1',
    #'MarkDown2',
    #'MarkDown3',
    #'MarkDown4',
    #'MarkDown5',
    'Size',
    #'Temperature',
    #'Unemployment',

    'md1_present',
    'md2_present',
    'md3_present',
    'md4_present',
    'md5_present',

    'IsHoliday_False',
    'IsHoliday_True',
    'Pre_christmas_no',
    'Pre_christmas_yes',
    'Black_Friday_no',
    'Black_Friday_yes',    
    'LaggedSales'
        ]


# ### Split data into training and test sets
# 
# Now we can split train test again and of course remove the trivial weekly sales data from the test set.

# In[33]:


train = df.iloc[:282451]
test = df.iloc[282451:]


# In[34]:


y = train['Difference'].values


# In[35]:


X = train[selector].values
train.head()


# ### Test - dev
# 
# Usually, model performance can be evaluated on the out-of-sample test set. However, since that data is not available, it may be wise to split our training set one more time in order to be able to test out of sample performance. Let's give up 20% of our training set for this sanity check development set.

# In[36]:


# Set seed for reproducability 
np.random.seed(42)
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Model selection
# 
# As usual, let's start off with all our imports.

# In[48]:


# Get Keras
# We will build a sequential model
from keras.models import Sequential
# Using fully connected layers
from keras.layers import Dense, Activation
# With vanilla gradient descent
from keras.optimizers import SGD
# Adam optimizer
from keras.optimizers import adam as adams
# Regulizer to avoid overfitting
from keras import regularizers 
# Dropout to avoid overfitting
from keras.layers import Dropout
from keras import optimizers


# Set some parameters to be used by all models

# In[38]:


learning_rate = 0.1
m,n = X.shape

# these hyper parameters can be tweaked after compiling the model
# this is useful for retraining an existing model under different params
batch_s = min(2**14, m//2)  # batch size: maxed at half size testset
print(n)
epochs = 10  # number of epochs per training round


# In[39]:


model = Sequential()
model.add(Dense(65, activation='relu', input_dim=n))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
adam=optimizers.Adam(lr=0.15, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mae')


# In[40]:


Sequential_feature_scaled = model.fit(X_train,y_train,batch_size=2048,epochs=epochs)


# ### Linear Regression
# 
# We will start with the baseline linear regression model to use as a benchmark for more complicated models. Note that the loss function is defined to reflect the mean average error.
# 
# We will store the output of each trained model in a history retrieval variable.

# In[41]:


lin_reg = Sequential()
lin_reg.add(Dense(1,input_dim=n))
lin_reg.compile(optimizer='adam', loss='mean_absolute_percentage_error')
history_lin_reg = lin_reg.fit(X_train,y_train,batch_size=batch_s,epochs=epochs,verbose=0)
lin_reg.evaluate(x=X_dev,y=y_dev)


# ### Tanh activation
# 
# We begin with a fairly simple model, include a hidden tanh activation layer. This model should already be able to outperfom our basic linear regression model.

# In[42]:


# Sequential model
tanh_model = Sequential()

# First hidden layer
tanh_model.add(Dense(32,activation='tanh',input_dim=n))

# Second hidden layer
tanh_model.add(Dense(16,activation='tanh'))

# Output layer
tanh_model.add(Dense(1,activation='sigmoid'))

# Compile the model
tanh_model.compile(optimizer=SGD(lr=learning_rate),
              loss='mean_squared_logarithmic_error',
              metrics=['acc'])

# Train
history_tanh = tanh_model.fit(X_train, y_train, # Train on training set
                         epochs=epochs, # We will train over 1,000 epochs
                         batch_size=batch_s, # Batch size 
                         verbose=0) # Suppress Keras output


# In[43]:


tanh_model.evaluate(x=X_dev,y=y_dev)


# ### Relu activation with momentum
# 
# In our next model, we replace the tanh function with the computationally more efficient Relu function. This should speed up the calculation of epochs and allow us to reduce our MSE faster. Furthermore, we will implement a moving average momentum element that smoothens out our gradient direction in its decent.

# In[44]:


# Sequential model
relu_momentum = Sequential()

# First hidden layer
relu_momentum.add(Dense(32,activation='relu',input_dim=n))

# Output layer
relu_momentum.add(Dense(1,activation='sigmoid'))

# Setup optimizer with learning rate of 0.01 and momentum (beta) of 0.9
momentum_optimizer = SGD(lr=learning_rate, momentum=0.9)

# Compile the model
relu_momentum.compile(loss='kullback_leibler_divergence', optimizer=momentum_optimizer)
# Train
history_relu_momentum = relu_momentum.fit(X_train, y_train, # Train on training set
                             epochs=epochs, # We will train over 1,000 epochs
                             batch_size=batch_s, # Batch size 
                             verbose=0) # Suppress Keras output
relu_momentum.evaluate(x=X_dev,y=y_dev)


# ### Adam optimizer with regularization
# 
# In our next model, we will stick with the relu activator, but replace the momentum with an Adam optimizer. Adaptive momumtum estimator uses exponentially weighted averages of the gradients to optimize its momentum.  However, since this method is known to overfit the model because of its fast decent, we will make use of a regulizer to avoid overfitting. The l2 regulizer adds the sum of absolute values of the weights to the loss function, thus discouraging large weights that overemphasize single observations.

# In[49]:


# Sequential model
adam_regularized = Sequential()

# First hidden layer now regularized
adam_regularized.add(Dense(32,activation='relu',
                input_dim=X_train.shape[1],
                kernel_regularizer = regularizers.l2(0.01)))

# Second hidden layer now regularized
adam_regularized.add(Dense(16,activation='relu',
                   kernel_regularizer = regularizers.l2(0.01)))

# Output layer stayed sigmoid
adam_regularized.add(Dense(1,activation='sigmoid'))

# Setup adam optimizer
adam_optimizer=adams(lr=learning_rate,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
adam_regularized.compile(optimizer=adam_optimizer,
              loss='poisson',
              metrics=['acc'])

# Train
history_adam_regularized=adam_regularized.fit(X_train, y_train, # Train on training set
                             epochs=epochs, # We will train over 1,000 epochs
                             batch_size=batch_s, # Batch size 
                             verbose=0) # Suppress Keras output
adam_regularized.evaluate(x=X_dev,y=y_dev)


# ### Tanh  optimizer with dropout
# 
# 

# In[51]:


# Sequential model
tanh_dropout = Sequential()

# First hidden layer
tanh_dropout.add(Dense(32,activation='relu',
                input_dim=X_train.shape[1]))

# Add dropout layer
tanh_dropout.add(Dropout(rate=0.5))

# Second hidden layer
tanh_dropout.add(Dense(16,activation='relu'))


# Add another dropout layer
tanh_dropout.add(Dropout(rate=0.5))

# Output layer stayed sigmoid
tanh_dropout.add(Dense(1,activation='sigmoid'))

# Setup adam optimizer
adam_optimizer=adams(lr=learning_rate,
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08)

# Compile the model
tanh_dropout.compile(optimizer=adam_optimizer,
              loss='cosine_proximity',
              metrics=['acc'])

# Train
history_dropout = tanh_dropout.fit(X_train, y_train, # Train on training set
                             epochs=epochs, # We will train over 1,000 epochs
                             batch_size=batch_s, # Batch size = training set size
                             verbose=0) # Suppress Keras output
tanh_dropout.evaluate(x=X_dev,y=y_dev)


# ### Model evaluation
#  
#  To evaluate the model, we will look at MAE and accuracy in terms of the number of times it correctly estimated an upward or downward deviation from the median.
# 

# In[52]:


plt.plot(history_lin_reg.history['loss'], label='Logistic regression')
plt.plot(history_tanh.history['loss'], label='Tanh Model')
plt.plot(history_relu_momentum.history['loss'], label='Relu Momentum')
#plt.plot(history_adam_regularized.history['loss'], label='Adam Regularized')
plt.plot(history_dropout.history['loss'], label='Tanh with Dropout')
plt.plot(Sequential_feature_scaled.history['loss'], label='New')

plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, cross_validation
from sklearn.metrics import mean_squared_error
import sklearn.metrics as skm
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from datetime import datetime
import timeit


# In[54]:


cv_l = cross_validation.KFold(len(X_train), n_folds=10, shuffle=True, random_state = 1)
regr = LassoCV(cv=cv_l, n_jobs = 2)


# In[55]:


regr = regr.fit( X_train, y_train )
y_pred = regr.predict(X_dev)


# In[56]:


from scipy import stats
y_pred_lin_reg = lin_reg.predict(X_dev,batch_size=batch_s)
y_pred_tanh_model = tanh_model.predict(X_dev,batch_size=batch_s)
y_pred_relu_momentum = relu_momentum.predict(X_dev,batch_size=batch_s)
y_pred_adam_regularized = adam_regularized.predict(X_dev,batch_size=batch_s)
y_pred_tanh_dropout = tanh_dropout.predict(X_dev,batch_size=batch_s)
y_pred = model.predict(X_dev,batch_size=batch_s)
stats.describe(y_pred_lin_reg)


# In[61]:


#http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

def plot_prediction(predicted,desciption):
    fig, ax = plt.subplots()
    ax.scatter(y_dev, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted by '+desciption)
    ax.plot([-30,30], [0,0], 'k-')   
    ax.plot([0,0], [-30,30], 'k-')
    plt.show()
    
plot_prediction(y_pred, 'new')
plot_prediction(y_pred_lin_reg, 'linear')
plot_prediction(y_pred_tanh_model, 'tanh')
plot_prediction(y_pred_relu_momentum, 'relu momentum')
plot_prediction(y_pred_adam_regularized, 'adam')
plot_prediction(y_pred_tanh_dropout,'dropout')
plot_prediction(y_pred,'lasso regression')


# Unfortunately, it appears that so far the models do not find any possible improvements over the median sales forecasts with the available explanatory variables.

# ## Forecasting sales
# 
# After we have created our model, we can predict things with it on the test set

# In[62]:


test.head()


# In[120]:


X_test = test.iloc[:,1:(n+1)].values
X_test.shape
test.head()
final_y_prediction = tanh_dropout.predict(X_test,batch_size=batch_s)


# To create the ids required for the submission we need the original test file one more time

# In[124]:


testfile = pd.read_csv('../input/test.csv')


# Let's add the means to our testfile and then subtract the expected difference.

# In[125]:


# Create final forecasts
testfile['prediction']=final_y_prediction
testfile['DateType'] = [datetime.strptime(date, '%Y-%m-%d').date() for date in testfile['Date'].astype(str).values.tolist()]
testfile['Month'] = [date.month for date in testfile['DateType']]
testfile['Month'] = 'Month_' + testfile['Month'].map(str)

testfile=testfile.merge(medians, how = 'outer', on = ['Type','Dept','Store','Month'])
testfile['prediction'].fillna(testfile['prediction'].median(), inplace=True) 
testfile['Median Sales'].fillna(testfile['Median Sales'].median(), inplace=True) 
testfile['prediction']+=testfile['Median Sales']
testfile.describe()


# Now we create the submission. Once you run the kernel you can download the submission from its outputs and upload it to the Kaggle InClass competition page.

# In[127]:


submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':testfile['prediction']})


# Check submission one more time

# In[128]:


submission.head()


# In[129]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




