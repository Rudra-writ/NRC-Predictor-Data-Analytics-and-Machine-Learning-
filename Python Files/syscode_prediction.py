from operator import index
import smtplib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import preprocessing
import sklearn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.callbacks import TensorBoard
from matplotlib import pyplot
from keras.optimizers import adam_v2
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from datetime import date
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pandas.plotting import autocorrelation_plot
import math
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset


np.set_printoptions(suppress = True)
pd.set_option('display.float_format', lambda x: '%0.5f' % x)
current_year = date.today().year


data = pd.read_excel('P:/master_yearly_total.xlsx')
data2 = pd.read_excel('P:/master_yearly_total.xlsx')
data.drop("Unnamed: 0", axis =1, inplace= True)
data2.drop("Unnamed: 0", axis =1, inplace= True)

data.drop(['system_code', 'master_total'], axis = 1, inplace = True)
labels = {'january':'sum', 'february': 'sum', 'march': 'sum', 'april': 'sum', 'may': 'sum', 'june':'sum', 'july':'sum', 'august':'sum', 'september':'sum', 'october':'sum', 'november':'sum', 'december': 'sum'}
data = data.groupby(['year'], as_index= False).agg(labels)
years = data['year'].unique()
data_narrowed =pd.DataFrame()
data_transposed = data.T
data_transposed = data_transposed.iloc[1:,:]
print(data.head())
print(data_transposed)



n_columns = data_transposed.shape[1]

col = data_transposed.iloc[:, 3]
print('.............................cols.........................................')
print(n_columns)
print(col)

print('-----------------------------')
for cols in range(0,(n_columns)):
    col = data_transposed.iloc[:, cols]
    col = pd.DataFrame(col)
    col.reset_index(level = 0, inplace = True)
   
    
    data_narrowed = data_narrowed.append(col)   

data_narrowed.reset_index(drop = True, inplace =True)
data_narrowed = data_narrowed.fillna(0)
data_narrowed['expenditure'] = data_narrowed.iloc[:, 0:].sum(axis = 1)
data_narrowed = data_narrowed[['index', 'expenditure']]

years = np.repeat(years,12)
years = pd.DataFrame({'years': years})
data_narrowed = pd.concat([data_narrowed, years], axis = 1)
data_narrowed['index' ] = data_narrowed['index'].str[:3]
data_narrowed['date'] = data_narrowed['index'].str.capitalize() + "-" + data_narrowed['years'].astype(str)

data_narrowed['date'] = pd.to_datetime(data_narrowed['date'])
data_narrowed = data_narrowed[['date', 'expenditure']]
data_narrowed = data_narrowed[data_narrowed['expenditure'] != 0]


data_narrowed.set_index('date',inplace=True)
print('data_narrowed....................................................................')
print(data_narrowed)
print("-----------------------------------------------------------------------------------------")



syscode_list = data2['system_code'].unique()
syscode_df = pd.DataFrame()
for syscode in syscode_list:
    
    dummy_df = data2.loc[data2['system_code'] == syscode]
    dummy_df.drop(['system_code', 'master_total'], axis = 1, inplace = True)
    years = data['year'].unique()
    process_df =pd.DataFrame()
    transposed_df = dummy_df.T
   
    transposed_df = transposed_df.iloc[1:,:]
    transposed_df.reset_index(drop = True, inplace = True)
    transposed_df.columns = range(transposed_df.columns.size)
    n_columns2 = transposed_df.shape[1]

    for cols in range(0,(n_columns2)):
        col2 = transposed_df.iloc[:, cols]
        col2 = pd.DataFrame(col2)
        col2.reset_index(level = 0, inplace = True)
        process_df = process_df.append(col2)  
    
    process_df.set_index('index', inplace = True)

    process_df.reset_index(drop = True, inplace =True)
    process_df = process_df.fillna(0)
    process_df[syscode] = process_df.iloc[:, 0:].sum(axis = 1)
    process_df = process_df[[ syscode]]
    
    syscode_df = pd.concat([syscode_df,process_df], axis = 1)
    process_df = process_df[0:0]



syscode_df_copy = pd.DataFrame()
data_narrowed.reset_index(drop=False, inplace = True)
syscode_df = syscode_df[0:36]
syscode_df = pd.concat([syscode_df, data_narrowed], axis = 1)
syscode_df = syscode_df.fillna(0)
syscode_df = syscode_df.set_index('date')
syscode_df_copy = syscode_df.copy(deep = True)


for syscode in syscode_list:
     syscode_df[syscode] =syscode_df[syscode].replace(0,syscode_df[syscode].mean() )





def adfuller_test(expenditure):
    result = adfuller(expenditure)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result,labels):
        print(label +' : '+str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

forecast_df = pd.DataFrame()

syscode_df['system_code_first_difference'] = syscode_df['FCLY-MCC'] - syscode_df['FCLY-MCC'].shift(1)
syscode_df = syscode_df.dropna()

adfuller_test(syscode_df['system_code_first_difference'])




future_dates=[syscode_df.index[-1] + DateOffset(months=x) for x in range(0,24)]


model2=sm.tsa.statespace.SARIMAX(syscode_df['FCLY-MCC'],order=(1, 1, 1),seasonal_order=(1,1,1,8), enforce_stationarity=False)
results=model2.fit()


future_dataset_df = pd.DataFrame(index=future_dates[1:], columns = syscode_df.columns)

future_df = pd.concat([syscode_df, future_dataset_df])

future_df['syscode_forecast']= results.predict(start=15,end= 59,dynamic=True)
future_df[future_df < 0] = 0
future_df[['FCLY-MCC','syscode_forecast']].plot(figsize=(10,8))

pyplot.show()





for syscode in syscode_list:
        model2=sm.tsa.statespace.SARIMAX(syscode_df[syscode],order=(1, 1, 1),seasonal_order=(1,1,1,8), enforce_stationarity=False)
        results=model2.fit()


        future_dataset_df = pd.DataFrame(index=future_dates[1:], columns = syscode_df.columns)

        future_df = pd.concat([syscode_df, future_dataset_df])

        future_df[syscode+'_forecast']= results.predict(start=15,end= 58,dynamic=True)
        future_df[future_df < 0] = 0
        forecast_df = pd.concat([forecast_df,future_df.iloc[:,-1]], axis = 1)
      
        future_df= future_df[0:0]
        future_dataset_df = future_dataset_df[0:0]



syscode_df = pd.concat([syscode_df,forecast_df], axis = 1)
syscode_df.drop(['expenditure'], axis = 1, inplace = True)
syscode_df[syscode_df < 0] = 0
print(syscode_df.iloc[:,0:10])
syscode_df.reset_index(drop=False, inplace = True)
syscode_df['index'] = syscode_df['index'].astype(str)




syscode_df_copy.drop([ 'expenditure'], axis = 1, inplace = True)



columns = list(syscode_df_copy.columns)
syscode_df_copy.reset_index(['date'], inplace = True)
print(syscode_df_copy.head(50))

syscode_df_copy2 = syscode_df

syscode_df_copy2.drop([ syscode for syscode in columns ], inplace = True, axis = 1 )
syscode_df_copy2.drop(['system_code_first_difference'], inplace =True,axis = 1 )
syscode_df_copy2.columns = syscode_df_copy2.columns.str.rstrip("_forecast")
syscode_df_copy2.columns = syscode_df_copy2.columns.str.replace('Managemen', 'Management')
syscode_df_copy2_new= syscode_df_copy2.rename(columns = {'index':'date'})
print(syscode_df_copy2_new.tail(50))


length= len(syscode_df_copy) -1
  

syscode_df_copy2_new = syscode_df_copy2_new.iloc[length:,:]
syscode_df_copy2_new.reset_index( drop = True, inplace = True)
syscode_df_copy.loc[:,'Type'] = 'Actual'
syscode_df_copy2_new.loc[:,'Type'] = 'SARIMAX'

print(syscode_df_copy2_new.tail(50))
syscode_df_copy['date'] = syscode_df_copy['date'].astype(str)
syscode_df_copy = pd.concat([syscode_df_copy,syscode_df_copy2_new], axis = 0 )
syscode_df_copy.reset_index(drop = True, inplace = True)
syscode_df_copy.to_excel("P:/data_dumped.xlsx")












