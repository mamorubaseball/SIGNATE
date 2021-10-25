#!/usr/bin/env python
# coding: utf-8

# # エラーで躓いたこと

# In[ ]:


df_h=groupby('hour').mean()
df_h.sort_values('bikes_available')

# sm.graphics.tsa.plot_acf(df['non_lack_of_bikes'],lags=7)
df['曜日']=df.index.strftime('%A')
df.groupby('曜日').mean()


# from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_scale = pd.DataFrame(scaler.fit_transform(df),index=df.index,columns=df.columns)

df_train= df[0:43801]
df_train=df[43801:]

