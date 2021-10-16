import pandas as pd


def make_cleaned_data():
    df=pd.read_csv('csv/status.csv')
    # citys=list(df['city'].unique())
    for i in range(69):
        station_id=i
        df.groupby('station_id').get_group(station_id)
        df.to_csv('csv/station/station_id{}.csv'.format(station_id),index=False)


    # df.groupby('city').get_group('')
def reduce_record():
    df=pd.read_csv('csv/station/station_id0.csv')
    df=df.columns[:1000]



def station0():
    df=pd.read_csv('csv/station/station_id0.csv')
    df.groupby(df.index//24).sum()


make_cleaned_data()