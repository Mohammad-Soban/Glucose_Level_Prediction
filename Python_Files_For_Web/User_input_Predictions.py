import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def preds_from_user_input(nfs, Reading_Given):
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df.set_index('Glucose_time', inplace=True)

    checked_df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['reading'] = scaler.fit_transform(df[['reading']])

    n_feat_list = []

    for i in range(2, nfs+1):
        if i % 2 == 0:
            n_feat_list.append(i)

    if nfs not in n_feat_list:
        n_feat_list.append(nfs)

    patience_lst = [10, 15, 20, 25]

    time_series_data = df['reading'].values

    best_parameter = {}
    best_rmse = float('inf')
    best_predictions = None
    best_model = None

    for n_features in n_feat_list:
        for pat in patience_lst:
            X, y = prepare_data(time_series_data, n_features)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            test_size = 18
            val_size = 24
            train_size = X.shape[0] + (n_features - 5) - test_size - val_size

            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

                # Print the shapes of the train, validation and test sets
            print("Train Shape: ", X_train.shape, y_train.shape)
            print("Validation Shape: ", X_val.shape, y_val.shape)
            print("Test Shape: ", X_test.shape, y_test.shape)

                
                # Building the LSTM Model
            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_features, 1)))
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
                best_parameter['n_features'] = n_features
                best_parameter['patience'] = pat
                best_predictions = predictions
                best_model = model

    print("*" * 50)
    print("Best Parameters: ", best_parameter)
    print("Best RMSE: ", best_rmse)
    print("Prediction :- ", best_predictions)

    validation_predictions_in_original_scale = scaler.inverse_transform(best_predictions)
    xlst = validation_predictions_in_original_scale.flatten()

    actual_values = df['reading'].values[train_size:train_size+val_size]
    actual_values = actual_values.reshape(-1, 1)
    actual_values_in_original_scale = scaler.inverse_transform(actual_values)
    actual_values_in_original_scale = actual_values_in_original_scale.flatten()

    final = pd.DataFrame()
    # Append the time of time series from values 240 to 264
    final['time'] = df.index[train_size:train_size+val_size]


    # Append the precited values with a shift of 1 and the predicted value at 240 being the mean of actual values at 237, 238 and 239. Append the mean first and then the predicted values
    final['Shifted_prediction'] = [np.mean(actual_values_in_original_scale[237:240])] + validation_predictions_in_original_scale.flatten().tolist()[:-1]
    final['unshifted_prediction'] = xlst.flatten().tolist()

    # Append the actual values
    final['actual'] = actual_values_in_original_scale.flatten()

    Reading_Given_scaled = scaler.transform(np.array(Reading_Given).reshape(-1, 1))

    n_features = best_parameter['n_features']

    X_new, _ = prepare_data(Reading_Given_scaled, n_features)

    # Handle the case where not enough data is available to create the required sequence
    if X_new.size == 0:
        # Pad the sequence with the initial readings (scaled)
        X_new = np.zeros((1, n_features, 1))
        X_new[0, :len(Reading_Given_scaled), 0] = Reading_Given_scaled.flatten()
    else:
        # Reshape the data to the format expected by the LSTM model
        X_new = X_new.reshape((X_new.shape[0], n_features, 1))

    # Ensure X_new is the correct shape for prediction
    if X_new.shape[0] != 1:
        X_new = X_new[-1].reshape((1, n_features, 1))

    predictions = []
    for i in range(10):
        pred = best_model.predict(X_new)
        predictions.append(pred[0, 0])  # Append the first element of the prediction
        # Update X_new by removing the first value and adding the new prediction
        X_new = np.append(X_new[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Inverse transform the predictions to get them back to the original scale
    predictions_in_original_scale = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions_in_original_scale = predictions_in_original_scale.flatten()

    return final, predictions_in_original_scale