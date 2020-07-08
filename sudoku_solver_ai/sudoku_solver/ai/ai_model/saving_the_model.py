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
from training_data import X_train2, y_train2


X_train, y_train = np.array(X_train2), np.array(y_train2)

