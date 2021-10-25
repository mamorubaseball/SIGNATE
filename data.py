#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import datetime
import xgboost as xgb
import pickle
import seaborn as sns


# In[13]:


def make_data():
    df=pd.read_csv('status.csv')
    train=df[df['predict']==0]
    test=df[df['predict']==1]
    #欠損値の除去；利用可能バイクの量が欠損している場合除去
    train.dropna(subset=['bikes_available'])
    train.to_csv('train.csv',index=False)
    test.to_csv('test.csv',index=False)

train=pd.read_csv('train.csv')
train=train.dropna(subset=['bikes_available'])
train.to_csv('train.csv',index=False)

    


# In[14]:


#データの可視化
#；ステーションIDの数
status = pd.read_csv("status.csv")
status["date"] = pd.to_datetime(status[["year", "month", "day", "hour"]])

pred_days = [
    "2014-04-01",
    "2014-04-02",
    "2014-04-03",
    "2014-04-04",
    "2014-04-05",
    "2014-04-06",
    "2014-04-07",
    "2014-08-01",
    "2014-08-02",
    "2014-08-03",
    "2014-08-04",
    "2014-08-05",
    "2014-08-06",
    "2014-08-07",
]


# In[15]:


import IPython


def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)

class CustomTimesSeriesSplitter:
    def __init__(self,pred_days,day_col='date'):
        self.day_col=day_col
        self.test_days=1
        self.pred_days=pred_days
        
    def split(self, X):
        for target_day in self.pred_days:
            target_day = pd.to_datetime(target_day)
            # train_start = X["date"].min()
            train_start = X["date"].min()
            train_end = X[X["date"] == target_day]["date"].iloc[0]  # - datetime.timedelta(days=1)
            val_start = X[X["date"] == target_day]["date"].iloc[0]  # - datetime.timedelta(days=1)
            val_end = X[X["date"] == target_day]["date"].iloc[0] + datetime.timedelta(days=1)
            # train_end =
            # val_start = X[X["date"] == target_day]["date"].iloc[0]

            train_mask = (train_start <= X["date"]) & (X["date"] <= train_end)
            val_mask = (val_start < X["date"]) & (X["date"] < val_end)

            yield X[train_mask].index.values, X[val_mask].index.values

    def get_n_splits(self):
        return len(self.pred_days)
cv=CustomTimesSeriesSplitter(pred_days)

    


# In[17]:


def show_cv_days(cv, X, dt_col):

    for ii, (tr, tt) in enumerate(cv.split(X)):
        print(f"----- Fold: ({ii + 1} / {cv.get_n_splits()}) -----")
        tr_start = X.iloc[tr][dt_col].min()
        tr_end = X.iloc[tr][dt_col].max()
        # tr_days = X.iloc[tr][day_col].max() - X.iloc[tr][day_col].min() + 1

        tt_start = X.iloc[tt][dt_col].min()
        tt_end = X.iloc[tt][dt_col].max()

        # tt_days = X.iloc[tt][day_col].max() - X.iloc[tt][day_col].min() + 1

        df = pd.DataFrame(
            {
                "start": [tr_start, tt_start],
                "end": [tr_end, tt_end],
                # "days": [tr_days, tt_days],
            },
            index=["train", "test"],
        )

        display(df)


def plot_cv_indices(cv, X, dt_col, lw=10):
    n_splits = cv.get_n_splits()
    _, ax = plt.subplots(figsize=(20, n_splits + 2))
    all_days = X[dt_col].unique().tolist()
    all_days = [pd.to_datetime(day) for day in all_days]

    memo = X[(X["station_id"] == 0) & (X["date"].isin(all_days))].index.values


    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        indices = indices[memo]

        # Visualize the results
        ax.scatter(
            X.loc[memo, dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    # plt.show()
    return ax

show_cv_days(cv, status, "date")


# In[18]:


plot_cv_indices(cv, status, "date")


# In[27]:


df=pd.read_csv('status.csv')
station_id=df['station_id'].unique()
station_df=pd.read_csv('station.csv')
lst=[]
for i in range(1,69):
    lst.append([station_df['lat'][i],station_df['long'][i]])
for i in lst:
    print(i[0],i[1])
    
    


# In[39]:


df=pd.read_csv('train.csv')
df_station=pd.read_csv('station.csv')

df_station['dock_count'][1]
df['dock_count']=0
for i in range(len(df)):
    si=df['station_id'][i]
    df['dock_count'][i]=df_station['dock_count'][si]
df.to_csv('train.csv',index=False)
    


# In[56]:


#時系列(横軸時間-台数)グラフの作成
import datetime

df=pd.read_csv('train.csv')
df_station=pd.read_csv('station.csv')
df['timedate']=0
df['day_bike']=0
df['month_day']=0
n=525658//24
for i in range(3):
    df['day_count']=df['dock_count'][i*24+1:(i+1)*24].sum()
    df['day_bike']=df['bikes_available'][i*24+1:(i+1)*24].sum()
    a=df['dock_count'][i*24+1:(i+1)*24].sum()
    year=df['year'][i*24+1]
    month=df['month'][i*24+1]
    day=df['day'][i*24+1]
    tdatetime=datetime.date(year,month,day)
    
df.to_csv('train_csv',index=False)
    
    


    
    

    


# In[ ]:





# In[ ]:




