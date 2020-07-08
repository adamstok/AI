import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers
import os
from sklearn.externals import joblib
from training_data import X_train, y_train


X_train, y_train = np.array(X_train), np.array(y_train)

x_train_reshaped = X_train.reshape(30,1,81)
y_train_reshaped = y_train.reshape(30,1,81)


regressor = Sequential()
regressor.add(LSTM(units = 1000, activation = 'relu', return_sequences = True, input_shape = ((x_train_reshaped.shape[1], 81))))
regressor.add(Dropout(0.3))
#regressor.add(LSTM(1200,dropout=0.0), return_sequences = True) # dropout=0.0
#regressor.add(Dropout(0.3))
regressor.add(Dense(400,activation='relu'))
regressor.add(Dense(100,activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(46,activation='relu'))
regressor.add(Dense(8,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=0.00010000)
#regressor.add(Dense(units=81))
regressor.add(Dense(1))
print(regressor.summary())
regressor.compile(optimizer=optimizer,loss='mean_squared_error')
regressor.fit(x_train_reshaped,y_train_reshaped,epochs=1000,batch_size=10)


# ??? model.h5 comming soon  ???
