import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def prepare_data(time_series_data, n_features):
    X, y = [], []
    for i in range(len(time_series_data)):
        end_ix = i + n_features
        if end_ix > len(time_series_data)-1:
            break
        seq_x, seq_y = time_series_data[i:end_ix], time_series_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)




def implement_LSTM_Model(df, n_features, n_epochs, train_size):
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    checked_df = df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['reading'] = scaler.fit_transform(df[['reading']])

    time_series_data = df['reading'].values

    X, y = prepare_data(time_series_data, n_features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    


