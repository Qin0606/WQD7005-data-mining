# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:29:51 2019

@author: leow.weiqin
"""

#Part 1 Data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Training set
df_train = pd.read_csv('goldprice_modified_train.csv')
training_set = df_train[['closing_price']] #can also use dataframe as this will be input into the scaler and
#converted into array

#array variable as per lecture (this is because keras use array as input - for X)
#training_set = df_train.iloc[:,1:2].values #.values change dataframe to arrays

#Feature Scaling
#RNN recommend to use normalization (min and max) instead of standardisation (mean and std)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
#scaler will transform data into array

#Need to specify a data structure to tell RNN the no of time steps and what to remember in RNN
#wrong number of time step could lead to overfitting or nonsense predictions

#Create a data structure with 60 timesteps and 1 output
# 60 timesteps means the RNN will refer to the previous 60 timestep to predict an output
#in our case, as our timestep is in days, the RNN will look at previous 60 days data
#The number 60 is by experiment, 1,20,30,40....60 turns out to be the best.
#1 month have roughly 20 financial day, thus 60 days = 3 previous month
X_train = [] #X-train, the input to RNN that contain the stock price of previous 60 previous timestep
y_train = [] #y_train, contain the stock price of the the current timestep

for i in range(2,1457):#can only start at the 60th data (days) as our timestep is 60 till the final data
    X_train.append(training_set_scaled[i-2:i,0]) # meaning data 0-59 will be used to predict price at 60
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train) 

#Reshaping - to add dimension to the x_train (add other features/ indicator beside the 'open' column)
#and to fulfill the input shape require by Keras

 #(batch size, timestep, no of indicator) in our case, 1
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Part 2 Build RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize
regressor = Sequential() #named as regressor as we are predicting continuous value

#Adding 1st LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1))) 
#units = no of neurons in lstm layers (too small units will not be able to capture the previous trend)
#return_sequence - True if using stacked LSTM
#For the last layer of LSTM, return_sequence = False
#input shape = (no of timestep, no of indicator) batch size or no of data is automatically taken into account

regressor.add(Dropout(rate=0.2))

#Adding 2nd LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate=0.2))

#Adding 3rd LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate=0.2))

#Adding 4th LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate=0.2))

#Add output layer
regressor.add(Dense(units=1))

#Compile 
regressor.compile(optimizer='adam',loss='mean_squared_error') #rmsprop recommended for RNN. But we try 'adam'

#Fit the RNN to training data
regressor.fit(X_train,y_train,batch_size=32,epochs=100)

#Part 3 Predict and visualize result

df_test = pd.read_csv('goldprice_modified_test.csv')
real_gold_price = df_test[['closing_price']]

#concatenate the training set and test set
df_total = pd.concat((df_train['closing_price'],df_test['closing_price']),axis=0)
inputs = df_total[(len(df_total) - len(df_test) - 60):].values
inputs = inputs.reshape(-1,1)
inputs_scaled = sc.transform(inputs)

X_test = []  

for i in range(2,367):
    X_test.append(inputs_scaled[i-2:i,0]) 
X_test= np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_gold_price, color='blue', label='Real Gold Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Gold Price')
plt.plot(training_set)
plt.legend()
plt.title('Real vs Predicted Google Stock Price')
plt.xlabel('time')
plt.ylabel('Google Stock Price')


