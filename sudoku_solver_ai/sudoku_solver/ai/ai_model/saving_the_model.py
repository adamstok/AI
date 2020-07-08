import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from training_data import X_train, y_train
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

X_train, y_train = np.array(X_train), np.array(y_train)

