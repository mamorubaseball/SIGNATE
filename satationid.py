#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
df=pd.read_csv('csv/status.csv')


def make_cleaned_data():
    df=pd.read_csv('csv/status.csv')
    # citys=list(df['city'].unique())
    for i in range(1):
        station_id=i
        df=df.groupby('station_id').get_group(station_id)
        df.to_csv('csv/station/station_id{}.csv'.format(station_id),index=False)
# make_cleaned_data()

    # df.groupby('city').get_group('')
def reduce_record():
    df=pd.read_csv('csv/station/station_id0.csv')
    df=df.columns[:1000]



def day_station0():
    df=pd.read_csv('csv/station/station_id0.csv')
    df['time']='0'
    df_copy=df.copy()
    for i,d in df.iterrows():
        d_time=datetime.datetime(d['year'],d['month'],d['day'],d['hour'])
#         print(time)
        df['time'][i]=d_time
#         df_copy.time=df.time.replace(i,d_time)
#         df_copy.time=df.time.where(df.time==i,d_time)
#         print(time)
        print(i)
    df.to_csv('csv/station/station_id0.csv',index=False)
    print('ok')

#     df[day_bike]=0
        
#     df.groupby(df.index//24).sum()
def plot_station():
    df=pd.read_csv('csv/station/station_id0.csv')
    x=df['time']
    y=df['bikes_available']
    plt.plot(x,y,label='bikes')
    plt.show()
# plot_station()


# In[14]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
# df=pd.read_csv('csv/status.csv')
# df['station_id'].unique()
for i in range(70):
    df=pd.read_csv('csv/status.csv')
    station_id=i
    df=df.groupby('station_id').get_group(station_id)
    df['timestamp']=pd.to_datetime(df[['year','month','day','hour']])
    df.to_csv('csv/station/station_id{}.csv'.format(station_id),index=False)
    print(station_id)

# df.to_csv('csv/station/station_id{}.csv'.format(station_id),index=False)
    


# In[41]:


get_ipython().system('pip install pydse')


# In[ ]:


#時系列解析といえば、ARIMAモデルの理解。
#pip install pydse
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
# print(sm.tsa.arma_order_select_ic(df['Passengers'], ic='aic', trend='nc'))
 
    
    


def model_station0():
    df=pd.read_csv('csv/station/station_id0.csv')
    arima_model = ARIMA(df['bikes_availabe], order=(4, 0, 2))
    res = arima_model.fit(dist=False)
    arima_predict = res.predict()
    plt.plot(df['bikes_availabe'], label='observation')
    plt.plot(arima_predict, '--', label='forcast')


# In[ ]:


df

