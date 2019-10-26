# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:41:25 2019

@author: Jor Yi
"""

import os
os.chdir('C:/Users/User/Desktop/WQD7005-data-mining-master/newscrawler')

import pandas as pd
import re
from textblob import TextBlob

# read data from csv file
news = pd.read_csv('news.csv')

#remove null rows and duplicate rows
news.dropna(inplace=True)
news.drop_duplicates(inplace=True)
news = news.reset_index(drop=True)

#transform DateTime column which contains not only date information
for i in (0,len(news)):
    if len(news['DateTime'][i]) in (9,10):
        continue
    else:
        new_date = re.search(r'\d{2}/\d{2}/\d{4}', news['DateTime'][i])
        news['DateTime'][i] = new_date.group()

#pd.to_datetime is able to read this format >> 01:06 AM10/18/2019 01:06:03 AM UTC-0400
news['DateTime'] = pd.to_datetime(news['DateTime'])

for i in range(0,len(news)):
    news['DateTime'][i] = news['DateTime'][i].date()

#filter down to only news about gold
for i in range(0,len(news)):
    if re.search(r'gold',news['news'][i],re.I) is not None:
        continue
    else:
        news.drop(i, inplace=True)

news.sort_values(by=['DateTime'],inplace=True)
news = news.reset_index(drop=True)

#derive sentiments in terms of polarity and subjectivity using textblob
news['polarity'] = ''
news['subjectivity'] = ''
news['polarity_description'] = ''

for i in range(0,len(news)):
    blob = TextBlob(news['news'][i])
    Sentiment = blob.sentiment
    news['polarity'][i] = Sentiment.polarity
    news['subjectivity'][i] = Sentiment.subjectivity
    
    if Sentiment.polarity > 0:
        news['polarity_description'][i] = 'Positive'
    elif Sentiment.polarity < 0:
        news['polarity_description'][i] = 'Negative'
    else:
        news['polarity_description'][i] = 'Neutral'
        
news['polarity'] = pd.to_numeric(news['polarity'])
news['DateTime'] = pd.to_datetime(news['DateTime'])

news.to_csv('news_preprocessed.csv',index=False)

polarity_mean = news.groupby('DateTime', as_index=False)['polarity'].mean()
polarity_mean.rename(columns={'polarity':'polarity_mean'},inplace=True)
polarity_mean.to_csv('polarity_mean.csv',index=False)