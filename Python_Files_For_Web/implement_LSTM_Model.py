import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
df.set_index('Glucose_time', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
df['reading'] = scaler.fit_transform(df[['reading']])
time_series_data = df['reading'].values

n_features_lst = [2, 4, 5, 6, 8, 10]
patience_lst = [5, 10, 15, 20]




def get_previous_3_values_mean(df, index):
    # Get the previous 3 values of the index
    previous_3_values = df.iloc[index-3:index]

    # Inverse transform the values
    previous_3_values = scaler.inverse_transform(previous_3_values)

    # Take the mean of the previous 3 values
    return np.mean(previous_3_values)


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


def Find_Best_Params_On_Validation_data():
    best_parameter = {}
    best_rmse = float('inf')
    best_predictions = None
    best_model = None

    for n in n_features_lst:
        for pat in patience_lst:
            X, y = prepare_data(time_series_data, n)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            train_size = 240
            val_size = 24

            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

            
            # Building the LSTM Model
            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

            # Fitting the model
            model.fit(X_train, y_train, epochs = 300, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

            # Choosing the best model based on the validation loss
            predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            if rmse < best_rmse:
                best_rmse = rmse
                best_parameter['n_features'] = n
                best_parameter['patience'] = pat
                best_predictions = predictions
                best_model = model


    validation_predictions = scaler.inverse_transform(best_predictions)

    print("*" * 50)
    print("Best Parameters: ", best_parameter)
    print("Best RMSE: ", best_rmse)
    print("Prediction :- ", best_predictions)
    return best_parameter, best_model, validation_predictions


def actual_preds_DF(validation_predictions):
    actual_values = df['reading'].values[-24:]
    actual_values = scaler.inverse_transform(actual_values.reshape(-1, 1))
    
    # Now I want it to be printed in the form of a dataframe with the predicted values with a shift of 1 and the actual values

    final = pd.DataFrame()
    # Append the time of time series from values 240 to 264
    final['Glucose_time'] = df.index[240: -24]

    # To get the mean of 240 pass the value of 240 in the get_previous_3_values_mean function
    mean_240 = get_previous_3_values_mean(df, 240)

    # Remove the last value of the predicted values and store it in a variable
    last_value = validation_predictions_in_original_scale[-1]
    validation_predictions_in_original_scale = validation_predictions_in_original_scale[:-1]

    xlst = validation_predictions.flatten()

    # Append the precited values with a shift of 1 and the predicted value at 240 being the mean of actual values at 237, 238 and 239. Append the mean first and then the predicted values
    final['Shifted_prediction'] = [mean_240] + validation_predictions_in_original_scale.flatten().tolist()
    final['unshifted_prediction'] = xlst.flatten().tolist()

    # Append the actual values
    final['actual'] = actual_values.flatten()

    return final