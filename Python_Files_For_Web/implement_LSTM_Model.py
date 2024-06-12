import pandas as pd
import numpy as np
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
scaler = MinMaxScaler(feature_range=(0, 1))



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

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    df['reading'] = scaler.fit_transform(df[['reading']])
    time_series_data = df['reading'].values

    n_features_lst = [2, 4, 6, 8, 10, 15]
    patience_lst = [10, 15, 20, 25]

    best_parameter = {}
    best_rmse = float('inf')
    best_predictions = None
    best_model = None

    for n in n_features_lst:
        for pat in patience_lst:
            X, y = prepare_data(time_series_data, n)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            print("From Validation Data : ", X.shape)

            test_size = 18
            val_size = 24
            train_size = X.shape[0] + (n - 5) - test_size - val_size

            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

            print(f"From Validation Data : X_train_Shape :  {X_train.shape}, X_Val_Shape : {X_val.shape}, X_test_Shape {X_test.shape}")
            print(f"From Validation Data : Y_train_Shape :  {y_train.shape}, Y_Val_Shape : {y_val.shape}, Y_test_Shape {y_test.shape}")
            
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

    # print("*" * 50)
    # print("Best Parameters: ", best_parameter)
    # print("Best RMSE: ", best_rmse)
    # print("Prediction :- ", validation_predictions)
    return best_parameter, best_model, validation_predictions



def find_best_params_test_data():

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    df['reading'] = scaler.fit_transform(df[['reading']])
    time_series_data = df['reading'].values

    n_features_lst = [2, 4, 6, 8, 10, 15]
    patience_lst = [10, 15, 20, 25]

    best_parameter = {}
    best_rmse = float('inf')
    best_predictions = None
    best_model = None

    for n in n_features_lst:
        for pat in patience_lst:
            X, y = prepare_data(time_series_data, n)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            train_size = X.shape[0] + (n - 5) - 18
            val_size = X.shape[0] - train_size
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            
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

    # print("*" * 50)
    # print("Best Parameters: ", best_parameter)
    # print("Best RMSE: ", best_rmse)
    # print("Prediction :- ", validation_predictions)
    return best_parameter, best_model, validation_predictions


def train_with_best_parameters(best_parameter):
    """
    This function will train the model on the entire dataset with the best parameters found on the validation dataset and the test dataset.
    """


    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    df['reading'] = scaler.fit_transform(df[['reading']])
    time_series_data = df['reading'].values

    X, y = prepare_data(time_series_data, best_parameter['n_features'])

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Splitting the data into train, validation and test sets
    # Train data should be of 264 values of the data set and not 240
    # Validation data is remaining 24 values of the data set

    train_size = X.shape[0] + (best_parameter['n_features'] - 5) - 18

    X_train, y_train = X[:train_size], y[:train_size]

    # Building the LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(best_parameter['n_features'], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fitting the model
    model.fit(X_train, y_train, epochs = 500, verbose=0)

    # Now we will predict the test data
    predictions = model.predict(X[train_size:])
    rmse = np.sqrt(mean_squared_error(y[train_size:], predictions))

    final_preds = scaler.inverse_transform(predictions).flatten()
    return final_preds, rmse, model



def lstm_on_entire_dataset(best_parameter, future_number):
    """
    This function will train the model on the entire dataset and predict the future values for the next 10 values
    """

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    df['reading'] = scaler.fit_transform(df[['reading']])
    time_series_data = df['reading'].values
    
    X, y = prepare_data(time_series_data, best_parameter['n_features'])

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Building the LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(best_parameter['n_features'], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fitting the model
    model.fit(X, y, epochs = 500, verbose=0)

    # Now we will predict the future 10 values.
    future_predictions = []
    last_value = X[-1]
    for i in range(future_number):
        future_predictions.append(model.predict(last_value.reshape(1, best_parameter['n_features'], 1))[0][0])
        last_value = np.append(last_value[1:], future_predictions[-1])

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()

    last_time = df.index[-1]
    
    future_time = pd.date_range(start=last_time, periods=future_number, freq='5min')[0:]
    future_df = pd.DataFrame()

    future_df['Glucose_time'] = future_time
    future_df['Prediction'] = future_predictions

    return future_df, model


def final_results():

    # Check whether the files already exist or not
    try:
        valid_final_preds = pd.read_csv("../CSV_Files/valid_predictions_lstm.csv")
        test_final_preds = pd.read_csv("../CSV_Files/test_predictions_lstm.csv")

        return valid_final_preds, test_final_preds

    except:
        # Now finding the best parameters on the validation dataset
        best_parameter_validation, best_model_validation, validation_preds = Find_Best_Params_On_Validation_data()
        
        # Save the parameters to a file
        best_parameter_validation_2 = pd.DataFrame(best_parameter_validation, index=[0])
        best_parameter_validation_2.to_csv("../Models/best_parameters_lstm.csv", index=False) 

        # Find the best parameters on the test data
        best_parameter_test, best_model_test, test_preds = find_best_params_test_data()

        # Save the parameters to a file
        best_parameter_test_2 = pd.DataFrame(best_parameter_test, index=[0])
        best_parameter_test_2.to_csv("../Models/best_parameters_test_lstm.csv", index=False)

        # Create a model on the entire dataset with the best validation dataset
        valid_final_preds, valid_model = lstm_on_entire_dataset(best_parameter_validation, 10)

        test_final_preds = lstm_on_entire_dataset(best_parameter_test, 10)

        # Saving the models into separate files for future use in a folder called Models
        valid_model.save("../Models/valid_model.h5")
        best_model_test.save("../Models/test_model.h5")

        # Save the valid_predictions to valid_predictions.csv and test_predictions to test_predictions.csv

        valid_final_preds.to_csv("../CSV_Files/valid_predictions_lstm.csv", index=False)
        test_final_preds[0].to_csv("../CSV_Files/test_predictions_lstm.csv", index=False)

    return valid_final_preds, test_final_preds[0]



def find_df_test_validation_predictions(best_param_validation, best_param_test):
    """
    This function will test the test predictions with the actual values and store that in a dataframe
    And also the same for the validation predictions
    """

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df.set_index('Glucose_time', inplace=True)

    # Training with the best parameters on the validation data
    validation_preds, rmse, model = train_with_best_parameters(best_param_validation)

    # Now we will find the actual values and the predictions for the validation data 