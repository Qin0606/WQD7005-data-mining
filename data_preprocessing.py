# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:05:18 2019

@author: leow.weiqin
"""

import pandas as pd
import numpy as np
from datetime import datetime

goldprice = pd.read_csv('goldprice.csv',thousands=',')



goldprice.info()
goldprice.describe()
len(goldprice)
goldprice['trend'] = ''

goldprice['date_text'] = pd.to_datetime(goldprice['date_text'])


#for i in range (len(goldprice)):
#    if i == 0:
#        goldprice['trend'][i] = ''
#    else:
#        if goldprice['closing_price'][i] > goldprice['closing_price'][i+1]:
#            goldprice['trend'][i] = 'upward'
#        elif goldprice['closing_price'][i]== goldprice['closing_price'][i-1]:
#            goldprice['trend'][i] = 'maintain'
#        else:
#            goldprice['trend'][i] = 'downward'

for i in range (1,len(goldprice)):
        if goldprice['closing_price'][i] > goldprice['closing_price'][i+1]:
            goldprice['trend'][i] = 'upward'
        elif goldprice['closing_price'][i]== goldprice['closing_price'][i+1]:
            goldprice['trend'][i] = 'maintain'
        else:
            goldprice['trend'][i] = 'downward'
            
goldprice.sort_values(by=['date_text'],inplace=True)
goldprice.dropna(inplace=True)

def get_day(x):
    return x.day_name()

def get_month(x):
    return x.month_name()

goldprice['day'] = goldprice['date_text'].apply(get_day)
goldprice['month'] = goldprice['date_text'].apply(get_month)

day_count = goldprice.groupby('day')['closing_price'].count()
day_mean = goldprice.groupby('day')['closing_price'].mean()
month_count = goldprice.groupby('month')['closing_price'].count()
month_mean = goldprice.groupby('month')['closing_price'].mean()

rint(month)