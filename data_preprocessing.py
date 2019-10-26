import os
os.chdir('C:/Users/tan.joryi/Desktop/p/data_mining/01. web scraping/WQD7005-data-mining-master/pricecrawler')

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

goldprice = pd.read_csv('goldprice.csv',thousands=',')

goldprice.info()
goldprice.describe()
len(goldprice)
goldprice['trend'] = ''

goldprice['date_text'] = pd.to_datetime(goldprice['date_text'])

goldprice.sort_values(by=['date_text'],ascending=False,inplace=True)
goldprice.dropna(inplace=True)

for i in range (1,len(goldprice)):
        if goldprice['closing_price'][i] > goldprice['closing_price'][i+1]:
            goldprice['trend'][i] = 'upward'
        elif goldprice['closing_price'][i]== goldprice['closing_price'][i+1]:
            goldprice['trend'][i] = 'maintain'
        else:
            goldprice['trend'][i] = 'downward'

goldprice['closing_at_daily_high'] = ''
goldprice['closing_at_daily_low'] = ''
goldprice['open_equals_closing'] = ''
goldprice['all_day_maintain'] = ''

for i in range (1,len(goldprice)+1):
    if goldprice['closing_price'][i] == goldprice['daily_high'][i]:
        goldprice['closing_at_daily_high'][i] = 1
    else:
        goldprice['closing_at_daily_high'][i] = 0
        
for i in range (1,len(goldprice)+1):
    if goldprice['closing_price'][i] == goldprice['daily_low'][i]:
        goldprice['closing_at_daily_low'][i] = 1
    else:
        goldprice['closing_at_daily_low'][i] = 0
        
for i in range (1,len(goldprice)+1):
    if goldprice['closing_price'][i] == goldprice['open_price'][i]:
        goldprice['open_equals_closing'][i] = 1
    else:
        goldprice['open_equals_closing'][i] = 0

for i in range (1,len(goldprice)+1):
    if goldprice['closing_price'][i] == goldprice['open_price'][i] == goldprice['daily_high'][i] == goldprice['daily_low'][i]:
        goldprice['all_day_maintain'][i] = 1
    else:
        goldprice['all_day_maintain'][i] = 0

for i in range (1,len(goldprice)-1):
    if goldprice['date_text'][i] - goldprice['date_text'][i+1] == timedelta(days=1):
        continue
    else:
        goldprice = goldprice.append({'date_text':goldprice['date_text'][i]-timedelta(days=1)},ignore_index=True)

goldprice.sort_values(by=['date_text'],inplace=True)
goldprice = goldprice.reset_index(drop=True)

def get_day(x):
    return x.day_name()

def get_day_no(x):
    return x.weekday()+1
    
def get_month(x):
    return x.month_name()

goldprice['day'] = goldprice['date_text'].apply(get_day)
goldprice['day_no'] = goldprice['date_text'].apply(get_day_no)
goldprice['month'] = goldprice['date_text'].apply(get_month)
goldprice['year'] = goldprice['date_text'].dt.year

goldprice = goldprice[['year','month','date_text','day','day_no','open_price','daily_low','daily_high','closing_price','trend','open_equals_closing','all_day_maintain','closing_at_daily_low','closing_at_daily_high']]
goldprice.to_csv('goldprice_preprocessed.csv',index=False)

#day_count = goldprice.groupby('day')['closing_price'].count()
#day_mean = goldprice.groupby('day')['closing_price'].mean()
#month_count = goldprice.groupby('month')['closing_price'].count()
#month_mean = goldprice.groupby('month')['closing_price'].mean()
#
#day_count.to_csv('day_count.csv')
#day_mean.to_csv('day_mean.csv')
#month_count.to_csv('month_count.csv')
#month_mean.to_csv('month_mean.csv')

for i in range(0,len(goldprice)):
    if np.isnan(goldprice['closing_price'][i]):
        goldprice['closing_price'][i] = np.mean([goldprice['closing_price'][i-1],goldprice['closing_price'][i+1]])
    if np.isnan(goldprice['open_price'][i]):
        goldprice['open_price'][i] = np.mean([goldprice['open_price'][i-1],goldprice['open_price'][i+1]])
    if np.isnan(goldprice['daily_low'][i]):
        goldprice['daily_low'][i] = np.mean([goldprice['daily_low'][i-1],goldprice['daily_low'][i+1]])
    if np.isnan(goldprice['daily_high'][i]):
        goldprice['daily_high'][i] = np.mean([goldprice['daily_high'][i-1],goldprice['daily_high'][i+1]])

get_polarity_mean = pd.read_csv('C:/Users/User/Desktop/WQD7005-data-mining-master/newscrawler/polarity_mean.csv')
get_polarity_mean['DateTime'] = pd.to_datetime(get_polarity_mean['DateTime'])

merge = pd.merge(get_polarity_mean,
                 goldprice[['date_text','trend']],
                 how='left',
                 left_on=['DateTime'],
                 right_on=['date_text'])


