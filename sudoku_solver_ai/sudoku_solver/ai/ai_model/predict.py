import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers
import joblib

regressor = tf.keras.models.load_model('./sudoku_model_v1.h5')
scaler = joblib.load('./scaler.save')

sudoku_test = (([0,0,3,0,0,0,6,0,0,0,0,0,1,0,5,0,0,0,1,0,0,0,2,0,0,0,5,0,5,0,8,0,7,0,6,0,0,0,9,0,0,0,4,0,0,0,2,0,5,0,4,0,9,0,7,0,0,0,6,0,0,0,3,0,0,0,4,0,2,0,0,0,0,0,2,0,0,0,9,0,0]))
sudoku_test = np.array(sudoku_test)
sudoku_test = sudoku_test.reshape(1,1,81)
sudoku_pred = regressor.predict(sudoku_test)
sudoku_pred = scaler.inverse_transform(sudoku_test)
print(sudoku_pred)
