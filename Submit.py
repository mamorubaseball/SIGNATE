#!/usr/bin/env python
# coding: utf-8

# # 提出ファイルの作成

# In[15]:


# station_0から70まで上の作業を行い、future/f_station_id0.csvで保存する
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
#いろいろインポート
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

all_df=pd.read_csv('csv/future_back/back_station_id0.csv',index_col='timestamp',parse_dates=True)
df=pd.read_csv('csv/future_back/back_station_id1.csv',index_col='timestamp',parse_dates=True)
print(len(df))
print(len(all_df))
al=all_df.append(df)
print(len(al))



# In[49]:


all_df=pd.read_csv('csv/future_back/back_station_id0.csv')

for i in range(1,70):
    df=pd.read_csv('csv/future_back/back_station_id{}.csv'.format(i))
    all_df=all_df.append(df)
    
df39=pd.read_csv('csv/future/fstation_id39.csv')
all_df.append(df39)
all_df=all_df.groupby('predict').get_group(1)

df=all_df
df.loc[df['bikes_available'] < 0, 'bikes_available']=-1*df.loc[df['bikes_available'] < 0, 'bikes_available'] 
df=df[['id','bikes_available']]
df=df.drop_duplicates(subset='id')
df.to_csv('csv/submit/submit1.csv',index=False,header=0)    


# In[41]:


df=pd.read_csv('csv/submit/submit2.csv')
df.loc[df['bikes_available'] < 0, 'bikes_available']=-1*df.loc[df['bikes_available'] < 0, 'bikes_available']  
df.to_csv('csv/submit/submit1.csv',index=False,header=0)


# In[ ]:




