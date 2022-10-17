#Discription: The program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predictthe closing stock price of a corporation using the past 60 day stock price.

#Import the libraries
import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get stock quote
df = web.DataReader('company', data_source='yahoo', start='2012-01-01')

#Visualize closing price hsitory
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Data')
plt.ylabel('Close Price in USD')
plt.show()

#Create a new dataframe with only the 'Close column'
data = df.filter(['Close'])
#Conver dataframe into numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

#Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create training data set and scaled training data set
train_data = scaled_data[0:training_data_len,:]
#Split data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()
#Convert x_train  and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
#Reshape data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Creating a test data set
test_data = scaled_data[training_data_len-60:,:]
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#Convert data into numpy array
x_test = np.array(x_test)
#Reshape data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get models predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Getting the quoted end price
company_quote = web.DataReader('company', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Create new dataframe
new_df = company_quote.filter(['Close'])
#Get the last 60 day closing prices in array
last_60_days = new_df[-60:].values
#Scale data to values b/w 0 to 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create empty list
X_test = []
#Append, conver to numpy and reshape past 60 days
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get predicted scaled price
pred_price = model.predict(X_test)
#Undo scaling
pred_price = scaler.inverse_transform(pred_price)

#Get quoted price
company_quote_official = web.DataReader('comapny', data_source='yahoo', start='2012-01-01', end='2019-12-17')
print(company_quote_official['Close'])
