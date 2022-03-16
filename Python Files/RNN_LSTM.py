from operator import index
import smtplib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import preprocessing
import sklearn
from sklearn.metrics import accuracy_score, r2_score
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
from sklearn.preprocessing import LabelEncoder, RobustScaler
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

# load dataset
np.set_printoptions(suppress = True)
pd.set_option('display.float_format', lambda x: '%0.5f' % x)



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
print(data_narrowed.shape)

train_size = int(len(data_narrowed) *0.85)
test_size = len(data_narrowed) - train_size
train, test = data_narrowed.iloc[0:train_size], data_narrowed.iloc[train_size : len(data_narrowed)]
print("train_test lengths")
print(len(train), len(test))

transformer = RobustScaler()
transformer = transformer.fit(train[['expenditure']])
train['expenditure'] = transformer.transform(train[['expenditure']])
test['expenditure'] = transformer.transform(test[['expenditure']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 3

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.expenditure, time_steps)
X_test, y_test = create_dataset(test, test.expenditure, time_steps)

print(X_train.shape, y_train.shape)
print(X_train)

model =keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units = 256,
      input_shape = (X_train.shape[1], X_train.shape[2]))))
model.add(keras.layers.Dense(units=128))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer= adam_v2.Adam(0.001))

history = model.fit(X_train, y_train, epochs=58, batch_size=1, validation_split=0.1, shuffle=False)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.show()
print( transformer.inverse_transform(X_test.reshape(-1,1)))

y_pred = model.predict(X_test)

y_train_inv = transformer.inverse_transform(y_train.reshape(-1,1))
y_test_inv = transformer.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = transformer.inverse_transform(y_test.reshape(-1,1))
print("Expected values-----------------------")
print(y_test_inv)
print("predicted Values----------------------")
print(y_pred_inv)


num_predictions = 5
dummy = pd.DataFrame()
dummy['expenditure'] = data_narrowed['expenditure']

for i in range(1,num_predictions):
    future_pred = dummy.iloc[-time_steps: ]
    future_pred_list = future_pred.values
    future_pred_list = future_pred_list.reshape(1,time_steps,1)

    print("-----------------------------------------------")
    future_prediction = abs(model.predict(future_pred_list))
    future_pred_list_inverse = transformer.inverse_transform(future_pred_list.reshape(1,-1))
    future_prediction_inverse = abs(transformer.inverse_transform(future_prediction))

    print(future_pred_list_inverse)
    
    print(future_prediction_inverse)

    future_dates=[dummy.index[-1] + DateOffset(months=1)]



    future_dataset_df = pd.DataFrame(index=future_dates[0:], columns = dummy.columns)
    print("future dataset df.........................")
    print(future_dataset_df)
    
    future_dataset_df = future_dataset_df.fillna(future_prediction.flatten()[0])
    dummy = pd.concat([dummy, future_dataset_df])
print(dummy)


tolerance_predict = 2
accuracy_predict = (np.abs( y_test_inv.flatten() - abs(y_pred_inv.flatten()))  < tolerance_predict).mean()
print(accuracy_predict)

print("Enter a System Code")
systemCode = input()
model2=sm.tsa.statespace.SARIMAX(data_narrowed[systemCode],order=(1, 1, 1),seasonal_order=(1,1,1,8))
results=model2.fit()

