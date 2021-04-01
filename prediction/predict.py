
# %%
import csv
from download import download
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.pylab as plt1
from matplotlib.pylab import rcParams
import seaborn as sns
from ipywidgets import interact
import datetime as dt

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv'
path_target = "./velo.csv"
download(url, path_target, replace=True)
data = pd.read_csv('velo.csv')
print(data)
# url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv"
# df = pd.read_csv(url)

# %%
data.columns = ['Date','Heure','Total cumulé','Total de la journée','Unnamed','Remarque']
data
# %%
data.drop([0,1])
bike = data.drop(columns=['Unnamed','Remarque'])
bike


# %%
bike_date = pd.to_datetime(bike['Date'] + ' ' + bike['Heure'], format='%d/%m/%Y %H:%M:%f')



# %%
bike_date

#%%
bike['Date'] = bike_date
bike
# %%
bike = bike.drop(columns='Heure')
bike
# %%
bike = bike.drop(columns='Total cumulé')
bike

# %%

bike09 = bike[(bike['Date'].dt.hour >= 0) & (bike['Date'].dt.hour <= 8)]
bike09tot = bike09.groupby(bike09['Date'].dt.date).sum()
bike09tot

# %%
bike09tot['Date'] = bike09tot.index
bike09tot['Date'] = pd.to_datetime(bike09tot['Date'])
bike09tot
# %%

bike09tot.drop(bike09tot[(bike09tot['Date'].dt.year == 2020) & (bike09tot['Date'].dt.month <= 5)].index, inplace=True)
bike09tot
# %%
bike09tot = bike09tot.drop(columns='Date')
bike09tot

# %%
plt1.xlabel("Date")
plt1.ylabel("Total de la journée")
plt1.plot(bike09tot)
# %%
# %%
import statsmodels
# %%
from statsmodels.tsa.stattools import adfuller
def test_stationnarity(dataset):

   
    dftest = adfuller(dataset,autolag = 'AIC')
    print("1. ADF :",dftest[0])
    print("2. pvalue :", dftest[1])
    print("3. Num of lags  :", dftest[2])
    print("4. Num of observations used for adf :", dftest[3])
    print("5. Critical values :")
    for key, val in dftest[4].items():
        print("\t",key,":", val)
    
# %%
test_stationnarity(bike09tot)
# %%
import pmdarima
# %%
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
# %%
stepwise_fit = auto_arima(bike09tot, trace=True,
                          suppress_warnings=True)
stepwise_fit.summary()                          
# %%
import statsmodels

# %%
from statsmodels.tsa.arima_model import ARIMA
# %%
print(bike09tot.shape)
train=bike09tot.iloc[:-30]
test=bike09tot.iloc[-30:]
print(train.shape,test.shape)
# %%
model=ARIMA(train,order=(0,0,2))
model=model.fit()
model.summary()
# %%
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels')
pred.index=bike09tot.index[start:end+1]
print(pred)
# %%
pred.plot(legend=True)
test.plot(legend=True)

# %%
test.mean()
pred.mean()
# %%
import sklearn.metrics
# %%
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test))
print(rmse)
# %%
model2=ARIMA(bike09tot,order=(0,0,2))
model2=model2.fit()
bike09tot.tail()
# %%
index_future_dates=pd.date_range(start='2021-03-30',end='2021-04-29')
pred=model2.predict(start=len(bike09tot),end=len(bike09tot)+30,typ='levels').rename('ARIMA Predictions')
pred.index=index_future_dates
print(pred)

