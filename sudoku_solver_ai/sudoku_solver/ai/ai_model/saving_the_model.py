mport pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers
import os
#from sklearn.externals import joblib
import joblib
from training_data import X_train2, y_train2



X_train, y_train = np.array(X_train2), np.array(y_train2)
x_train_reshaped = X_train.reshape(30,1,81)
y_train_reshaped = y_train.reshape(30,1,81)



regressor = Sequential()
regressor.add(LSTM(units = 1000, activation = 'relu', return_sequences = True, input_shape = ((x_train_reshaped.shape[1], 81))))
regressor.add(Dropout(0.3))
regressor.add(Dense(1200,activation='relu'))
regressor.add(Dense(800,activation='relu'))
regressor.add(Dropout(0.2))
regressor.add(Dense(400,activation='relu'))
regressor.add(Dense(100,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=0.00010000)
regressor.add(Dense(units=81))
print(regressor.summary())
regressor.compile(optimizer=optimizer,loss='mean_squared_error')
regressor.fit(x_train_reshaped,y_train_reshaped,epochs=10000,batch_size=10)
regressor.save('./sudoku_model_v1.h5')

# ??? model.h5 comming soon  ???









































