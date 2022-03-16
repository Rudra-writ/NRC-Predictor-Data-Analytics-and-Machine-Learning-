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
'''data = pd.read_excel('P:/ai_trainer_data2.xlsx')
data.drop("Unnamed: 0", axis =1, inplace= True)
data_predict = data.loc[data['year']== current_year]
data_train = data.loc[~(data['year'] == current_year)]
labels = pd.DataFrame()
labels_predict = pd.DataFrame()


data_train = pd.concat([data_train]* 30, ignore_index= True)
data_train = data_train.sample(frac = 1)



labels['master_total'] =data_train.loc[:, 'master_total']

features = data_train.iloc[:,0:39]

le=preprocessing.LabelEncoder()
features['system_code'] = le.fit_transform(features['system_code'])
features = features.drop(['year'], axis = 1)

features.loc[:,'january_po': 'december_mt'] = MinMaxScaler().fit_transform(features.loc[:,'january_po': 'december_mt'])
features.reset_index(drop = True, inplace = True)
labels.reset_index(drop = True, inplace = True)



X = features.loc[:,'system_code':'completion_rate'].values
y = labels['master_total'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



model = Sequential()
model.add(Dense(256, activation = 'relu', input_dim = 38))
model.add(Dense(128, activation ='relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense (1, activation = "linear"))

#model training

model.compile( loss = 'mean_squared_error', optimizer= adam_v2.Adam(learning_rate = 0.001) , metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 5, batch_size = 1, verbose =1)
model.save('my_model2.h5')

y_pred = model.predict(X_test)
print(y_test[0:50])
print(y_pred[0:50])
tolerance = 5000
accuracy = (np.abs(y_test - abs(y_pred))  < tolerance).mean()
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

print("Overall Accuracy:")
print(round(accuracy,2) * 100 )'''



#Predictions for the desired dataset begins here
'''labels_predict['master_total'] = data_predict.loc[:, 'master_total']

features_predict = data_predict.iloc[:,0:39]

le=preprocessing.LabelEncoder()
features_predict['system_code'] = le.fit_transform(features_predict['system_code'])
features_predict = features_predict.drop(['year'], axis = 1)

features_predict.loc[:,'january_po': 'december_mt'] = MinMaxScaler().fit_transform(features_predict.loc[:,'january_po': 'december_mt'])
features_predict.reset_index(drop = True, inplace = True)
labels_predict.reset_index(drop = True, inplace = True)

X_predict = features_predict.loc[:,'system_code':'completion_rate'].values
y_predict = labels_predict['master_total'].values

my_model = load_model('my_model2.h5')

y_predict_ai = my_model.predict(X_predict)

print(y_predict[0:50])
print(y_predict_ai[0:50])

tolerance_predict = 5000
accuracy_predict = (np.abs(y_predict - abs(y_predict_ai))  < tolerance_predict).mean()

error = (100 - accuracy_predict)/100

mse = sklearn.metrics.mean_squared_error(y_predict, y_predict_ai)
rmse = math.sqrt(mse)


score_predict = my_model.evaluate(X_predict, y_predict,verbose=1)
print(score_predict)
print(rmse)
print("error:")
print (error)

print("Overall Accuracy:")
print(round(accuracy_predict,2) * 100 )

features_predict['system_code'] = le.inverse_transform(features_predict['system_code'])
features_predict = features_predict['system_code']
predictions =pd.DataFrame(y_predict_ai, columns = ['predictions'])
final_df = pd.concat([features_predict, predictions], axis = 1)

print(final_df.head(50))'''

data = pd.read_excel('P:/master_yearly_total.xlsx')
data.drop("Unnamed: 0", axis =1, inplace= True)
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

print(data_narrowed)



test_result = adfuller(data_narrowed['expenditure'])

def adfuller_test(expenditure):
    result = adfuller(expenditure)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result,labels):
        print(label +' : '+str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(data_narrowed['expenditure'])

data_narrowed['expenditure_first_difference'] = data_narrowed['expenditure'] - data_narrowed['expenditure'].shift(3)
print(data_narrowed)
data_narrowed = data_narrowed.dropna()

adfuller_test(data_narrowed['expenditure_first_difference'])


data_narrowed['expenditure_first_difference'].plot()


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data_narrowed['expenditure_first_difference'].iloc[2:],lags=14,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(data_narrowed['expenditure_first_difference'].iloc[2:],lags=14,ax=ax2)

#pyplot.show()

#p = 1, d= 1, q = 1

model = ARIMA(data_narrowed['expenditure'], order=(1,1,1))
model_fit = model.fit()

data_narrowed['forecast']=model_fit.predict(start=15,end=31,dynamic=True)
data_narrowed[['expenditure','forecast']].plot(figsize=(12,8))

pyplot.show()


model2=sm.tsa.statespace.SARIMAX(data_narrowed['expenditure'],order=(1, 1, 1),seasonal_order=(1,1,1,8))
results=model2.fit()


data_narrowed['forecast']= results.predict(start=15,end=40,dynamic=True)
data_narrowed[['expenditure','forecast']].plot(figsize=(12,8))

pyplot.show()

future_dates=[data_narrowed.index[-1] + DateOffset(months=x) for x in range(0,24)]



future_dataset_df = pd.DataFrame(index=future_dates[1:], columns = data_narrowed.columns)
print("future dataset df.........................")
print(future_dataset_df)

future_df = pd.concat([data_narrowed, future_dataset_df])

future_df['forecast']= results.predict(start=15,end= 56,dynamic=True)
future_df[['expenditure','forecast']].plot(figsize=(12,8))
print(future_df)
pyplot.show()
expenditure_df = pd.DataFrame(columns = future_df.columns)
future_forecast_df = pd.DataFrame(columns = future_df.columns)
current_date = date.today()
future_df = future_df.fillna(0)

expenditure_df = future_df.loc[future_df['expenditure'] != 0]
future_forecast_df = future_df.loc[future_df['expenditure'] == 0]



print(expenditure_df)
print(future_forecast_df)
real_sum = expenditure_df['expenditure'].sum()
forecast_sum = future_forecast_df['forecast'].sum()

total_expenditure = real_sum + forecast_sum

print("The total Expenditure for the launch vehicle would be:" + " " +str(total_expenditure))
data_narrowed = data_narrowed.fillna(0)
util_df = data_narrowed.loc[data_narrowed['forecast'] != 0]
exp_actual_list = util_df['expenditure'].values


exp_predicted_list = util_df['forecast'].values



print(data_narrowed)

print(exp_actual_list)

print(exp_predicted_list)
util_df.reset_index( drop = False, inplace = True)

future_df = future_df.loc[future_df['expenditure'] == 0]
future_df.reset_index(drop = False, inplace = True)
print(future_df)

tolerance_predict = 300000
accuracy_predict = (np.abs( exp_predicted_list[:17] - abs(exp_actual_list[:17]))  < tolerance_predict).mean()

print("accuracy is: "+str(accuracy_predict))


   




















