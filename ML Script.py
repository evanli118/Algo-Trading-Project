#!/usr/bin/env python
# coding: utf-8

# ## Data Collection/Organization/Processing

# Collecting Data From API

# In[1]:


pip install finnhub-python


# API documentation: https://finnhub.io/docs/api/introduction 

# In[2]:


import finnhub
import numpy as np
import pandas as pd
import requests    
import math
from secrets import API_token  #importing my API_token
import time
import datetime
from datetime import date


# In[3]:


weights = []
parameters = []
y = []


# In[4]:


payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = payload[0]
symbols = df['Symbol'].values.tolist()
len(symbols)


# In[5]:


def chunk(lst, n):
    #give successive n-sized chunks from a list
    for i in range (0, len(lst), n):
        yield lst[i : i + n]


# In[6]:


symbol_lists = list(chunk(symbols, 10))


# In[7]:


finnhub_client = finnhub.Client(api_key=API_token)


# In[8]:


def get_single_stock_data(symbol):
    
    # basic_profile = finnhub_client.company_profile2(symbol=symbol) #dict, doesn't seem to be useful
    # quote = finnhub_client.quote(symbol) #dict, doens't seem to useful because the timeframe of data provided is too short
    basic_fin = finnhub_client.company_basic_financials(symbol, 'all')  #dict:basic financial info of a company, >1 layer
    #news_sentiment = finnhub_client.news_sentiment(symbol) #dict: >1 layer; currently the API has taken it down
    #peers = finnhub_client.company_peers(symbol) #list of string, may or may not need
    rec_trends = finnhub_client.recommendation_trends(symbol) #dict, needs to choose period
    #candles = finnhub_client.stock_candles(symbol, 'D', 1590988249, 1591852249) #dict, needs to choose UNIX period and resolution; doesn't seem useful
    pattern_rec = finnhub_client.pattern_recognition(symbol, 'M') #dict, pattern of a month, needs to refine data
    sup_res = finnhub_client.support_resistance(symbol, 'M') #dict with 1 key only
    aggregate_indicators = finnhub_client.aggregate_indicator(symbol, 'M') #dict with >1 layer, needs to refine data
    #techical_indicators: see dedicated code segment below
    social_sentiment = finnhub_client.stock_social_sentiment(symbol) #dict, data is messy, can add in further iterations
    
    #extracting y-variable first
    #y.append(basic_fin['metric']['13WeekPriceReturnDaily'])

    #breaking down basic_fin and extracting info
    basic_fin['metric'].pop('52WeekHighDate')
    basic_fin['metric'].pop('52WeekLowDate')
    basic_fin['metric'].pop('13WeekPriceReturnDaily')
    basic_fin['metric'].pop('26WeekPriceReturnDaily')
    basic_fin['metric'].pop('52WeekPriceReturnDaily')
    single_stock_param = []
    for key in basic_fin['metric'].keys():
        single_stock_param.append(basic_fin['metric'][key])

#     for key in basic_fin['series']['annual'].keys():     #this seems to be troublesome because not all companies have the same data
#         for i in range(2):
#             single_stock_param.append(basic_fin['series']['annual'][key][i]['v']) #extracting the value that's nested at the most inner layer of the dict

    #extracting useful data from news sentiment; currently the API has taken it down
#     single_stock_param.append(news_sentiment['companyNewsScore']-news_sentiment['sectorAverageNewsScore'])
#     single_stock_param.append(news_sentiment['buzz']['weeklyAverage'])
#     single_stock_param.append(news_sentiment['sentiment']['bearishPercent'])
#     single_stock_param.append(news_sentiment['sentiment']['bullishPercent'])

    #extracting useful data from recommended trends
    average_buy = 0
    average_hold = 0
    average_sell = 0
    average_strong_buy = 0
    average_strong_sell = 0
    for i in range(11):
        average_buy += rec_trends[i]['buy']
        average_hold += rec_trends[i]['hold']
        average_sell += rec_trends[i]['sell']
        average_strong_buy += rec_trends[i]['strongBuy']
        average_strong_sell += rec_trends[i]['strongSell']
    single_stock_param.append(average_buy / 12)
    single_stock_param.append(average_hold / 12)
    single_stock_param.append(average_sell / 12)
    single_stock_param.append(average_strong_buy / 12)
    single_stock_param.append(average_strong_sell / 12)

    #extracting useful data from support/resistance 
    sup_res_lst = []
    for value in sup_res['levels']:
        sup_res_lst.append(value)
    lst_range = len(sup_res_lst)-4
    for i in range (lst_range):  #this is to handle the difference in the number of levels returned. The baseline is 4
        sup_res_lst.pop()
    single_stock_param.append(sup_res_lst)
    
    #extracting useful info from aggregate indicators
    signal = 0
    if aggregate_indicators['technicalAnalysis']['signal'] == 'buy':
        signal = 10
    elif aggregate_indicators['technicalAnalysis']['signal'] == 'sell':
        signal = -10

    single_stock_param.append(signal)
    single_stock_param.append(aggregate_indicators['trend']['adx'])

    #now, taking a crack at technical analysis indicators. We will only use 2 indicators for now
    desired_window = (date.today()).strftime("%d/%m/%Y")
    desired_unix_3months = int(time.mktime(datetime.datetime.strptime(desired_window, "%d/%m/%Y").timetuple()) - 7884000*2)
    desired_unix_now = int(time.mktime(datetime.datetime.strptime(desired_window, "%d/%m/%Y").timetuple())-24*60*60)

    MACD = finnhub_client.technical_indicator(symbol=symbol, resolution='D', _from=desired_unix_3months, to=desired_unix_now, indicator='macd')
    for value in MACD['macdSignal']:
        single_stock_param.append(value)
    RSI = finnhub_client.technical_indicator(symbol=symbol, resolution='D', _from=desired_unix_3months, to=desired_unix_now, indicator='rsi')
    for value in RSI['rsi']:
        single_stock_param.append(value)
        

    single_stock_param = np.array(single_stock_param, dtype=object)
    
    return single_stock_param


# In[9]:


def get_y (symbol):
    basic_fin = finnhub_client.company_basic_financials(symbol, 'all')
    y.append(basic_fin['metric']['13WeekPriceReturnDaily'])


# In[10]:


for symbol_list in symbol_lists:
    for symbol in symbol_list:  #this is to deal with the 60api calls/min limit
        try:
            parameters.append(get_single_stock_data(symbol))
        except (RuntimeError, TypeError, KeyError, ValueError, IndexError):
            print(symbol + " has insufficient data")
        time.sleep(10)


# In[11]:


parameters = np.array(parameters)
parameters = np.where(parameters==None, 0, parameters)
parameters.shape


# In[12]:


for symbol_list in symbol_lists:
    for symbol in symbol_list:  #this is to deal with the 60api calls/min limit
        try:
            get_y(symbol)
        except (RuntimeError, TypeError, KeyError, ValueError, IndexError):
            print(symbol + " has insufficient data")
        time.sleep(1)


# In[14]:


#remove1 = symbols.index('BRK-B')
remove2 = symbols.index('OGN')


# In[15]:


y = np.array(y)
#y = np.delete(y, remove1)
y = np.delete(y, remove2)
y_nones = np.where(y == None)
y = np.delete(y,y_nones[0][0])
y.shape


# In[16]:


copy_p = parameters


# In[17]:


parameters = np.delete(parameters, y_nones[0][0], 0)
parameters.shape


# In[18]:


#writing the now cleaned up parameters to file
p_file = open("parameters.txt", "w")
for row in parameters:
    np.savetxt(p_file, row, fmt = '%s')
p_file.close()


# In[23]:


#reading from the file if parameters are already in the file
#parameters = np.loadtxt("parameters.txt").reshape(502, 386)


# In[24]:


#writing the now cleaned up y_parameters to file
y_file = open("y.txt", "w")
np.savetxt(y_file, y, fmt = '%s')
y_file.close()


# In[ ]:


#reading from the file if y_parameters are already in the file
#y = np.loadtxt("y.txt").reshape(502,)

