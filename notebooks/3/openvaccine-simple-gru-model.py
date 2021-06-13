#!/usr/bin/env python
# coding: utf-8

# This notebook shows you how to build a model for predicting degradation at various locations along RNA sequence. 
# * We will first pre-process and tokenize the sequence, secondary structure and loop type. 
# * Then, we will use all the information to train a model on degradations recorded by the researchers from OpenVaccine. 
# * Finally, we run our model on the public test set (shorter sequences) and the private test set (longer sequences), and submit the predictions.
# 
# For more details, please see [this discussion post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303).
# 
# ---
# 
# Updates:
# 
# * V7: Updated kernel initializer, embedding dimension, and added spatial dropout. Changed epochs and validation split. All based on [Tucker's excellent kernel](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm#Training) (go give them an upvote!).
# * V8-9: Changed loss from MSE to M-MCRMSE, which is the official competition metric. See [this post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211) for more details. Changed number of epochs to 100.
# * V10: loss function: M-MCRMSE -> MCRMSE
# * V11: Filter `signal_to_noise` to be greater than 1, since the same was applied to private set. See [this post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992). 
# * V12: Loss function: MCRMSE -> M-MCRMSE.
# * V13: Decrease and stratify validation set based on `SN_filter`. Increase embedding size, GRU dimensions, number of layers. All inspired from Tucker's work again.

# In[1]:


import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ## Set seed to ensure reproducibility

# In[2]:


tf.random.set_seed(2020)
np.random.seed(2020)


# ## Helper functions and useful variables

# In[3]:


# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']


# In[4]:


y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))


# In[5]:


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


# In[6]:


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))


# In[7]:


def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5, 
                sp_dropout=0.2, embed_dim=200, hidden_dim=256, n_layers=3):
    inputs = L.Input(shape=(seq_len, 3))
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])
    )
    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)
    
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    # Since we are only making predictions on the first part of each sequence, 
    # we have to truncate it
    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation='linear')(truncated)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    
    return model


# In[8]:


def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


# In[9]:


def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )


# ## Load and preprocess data

# In[10]:


data_dir = '/kaggle/input/stanford-covid-vaccine/'
train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
sample_df = pd.read_csv(data_dir + 'sample_submission.csv')


# In[11]:


train = train.query("signal_to_noise >= 1")


# In[12]:


# We will use this dictionary to map each character to an integer
# so that it can be used as an input in keras
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])


# In[13]:


x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.1, random_state=34, stratify=train.SN_filter)


# Public and private sets have different sequence lengths, so we will preprocess them separately and load models of different tensor shapes.

# In[14]:


public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df, token2int)
private_inputs = preprocess_inputs(private_df, token2int)


# ## Build and train model
# 
# We will train a bi-directional GRU model. It has three layer and has dropout. To learn more about RNNs, LSTM and GRU, please see [this blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

# In[15]:


model = build_model(embed_size=len(token2int))
model.summary()


# In[16]:


history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=75,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5')
    ]
)


# ## Evaluate training history
# 
# Let's use Plotly to quickly visualize the training and validation loss throughout the epochs.

# In[17]:


fig = px.line(
    history.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'MCRMSE'}, 
    title='Training History')
fig.show()


# ## Load models and make predictions

# Public and private sets have different sequence lengths, so we will preprocess them separately and load models of different tensor shapes. This is possible because RNN models can accept sequences of varying lengths as inputs.

# In[18]:


# Caveat: The prediction format requires the output to be the same length as the input,
# although it's not the case for the training data.
model_public = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))
model_private = build_model(seq_len=130, pred_len=130, embed_size=len(token2int))

model_public.load_weights('model.h5')
model_private.load_weights('model.h5')


# In[19]:


public_preds = model_public.predict(public_inputs)
private_preds = model_private.predict(private_inputs)


# ## Post-processing and submit

# For each sample, we take the predicted tensors of shape (107, 5) or (130, 5), and convert them to the long format (i.e. $629 \times 107, 5$ or $3005 \times 130, 5$):

# In[20]:


preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head()


# In[21]:


submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)

