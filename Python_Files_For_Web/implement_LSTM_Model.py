import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import tensorflow as tfr
import warnings

warnings.filterwarnings("ignore")
scaler = MinMaxScaler(feature_range=(0, 1))

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

    n_features_lst = [4, 5, 6, 8, 10, 15, 18]
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

    n_features_lst = [4, 5, 6, 8, 10, 15, 18]
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


def find_df_test_validation_predictions(best_param_validation, best_param_test):
    """
    This function will test the test predictions with the actual values and store that in a dataframe
    And also the same for the validation predictions
    """

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df.set_index('Glucose_time', inplace=True)

    # Training with the best parameters on the validation data
    validation_preds, valid_rmse, valid_model = train_with_best_parameters(best_param_validation)
    # Now we will create a dataframe with the actual values and the predictions

    df_validation = pd.DataFrame()
    # Shape of the predictions
    preds_shape = validation_preds.shape[0]
    df_validation['Actual'] = df['reading'][-preds_shape:]
    df_validation['Predictions'] = validation_preds

    
    # Training with the best test parameters
    test_preds, test_rmse, test_model = train_with_best_parameters(best_param_test)
    df_test = pd.DataFrame()
    
    # Shape of the predictions
    preds_shape = test_preds.shape[0]
    df_test['Actual'] = df['reading'][-preds_shape:]
    df_test['Predictions'] = test_preds

    return df_validation, df_test, valid_rmse, test_rmse


def lr_schedule(epoch, lr):
    return lr * 0.995



# Function to find the best parameters and the best model using CNN and LSTM
def best_params_final():
    """
        This function will give me the best 3 parameters and best 3 models using Bidirectional and Unidirectional LSTM Layer, Conv1D Layer, Dense Layer, MaxPooling Layer and Dropout rate of 0.2 and try to test the model on the last 20 values of the dataset.
    """
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    
    # Drop all the columns which have unnamed in them
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # If reading_time is present in the dataset, then convert it to a proper date time format
    if 'Glucose_time' not in df.columns:
        df['reading_time'] = pd.to_datetime(df['reading_time'], unit='ms')
        df['reading_time'] = pd.to_datetime(df['reading_time'], format='%Y-%m-%d %H:%M:%S')
        df = df.rename(columns={'reading_time': 'Glucose_time'})
        df = df.sort_values(by='Glucose_time')

    # Set the Glucose_time to datetime format and set it as the index
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    checked_df = df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['reading'] = scaler.fit_transform(df[['reading']])

    time_series_data = df['reading'].values

    n_features_lst = [3, 18, 19, 20]
    batch_size_lst = [16, 32]
    patience_lst = [25]

    best_parameters = []
    best_predictions = []
    best_models = []
    results_sheet = pd.DataFrame()
    rmse_list = []

    for n_feat in n_features_lst:
        for batch_size in batch_size_lst:
            for patience in patience_lst:
                print(f"Training model with n_features={n_feat}, dropout_rate=0.2, batch_size={batch_size}, patience={patience}")

                # Prepare data
                train_data_scaled = time_series_data[:-20]
                test_data_scaled = time_series_data[-(n_feat + 20):]

                X_train, y_train = prepare_data(train_data_scaled, n_feat)
                X_test, y_test = prepare_data(test_data_scaled, n_feat)

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # Define the model
                model = Sequential()
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_feat, 1)))
                if n_feat > 3:
                    model.add(MaxPooling1D(pool_size=2))
                elif n_feat > 2:
                    model.add(MaxPooling1D(pool_size=1))
                model.add(Dropout(0.2))

                model.add(LSTM(units=200, activation='tanh', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Bidirectional(LSTM(units=150, activation='relu', return_sequences=True)))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True)))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(LSTM(units=50, activation='relu', return_sequences=False))  # Set return_sequences to False
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Dense(100, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))

                # Compile the model
                optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
                model.compile(optimizer=optimizer, loss='mse')

                early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
                lr_scheduler = LearningRateScheduler(lr_schedule)

                # Train the model
                history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stop, lr_scheduler])

                predictions = []
                curr_sequence = X_test[0].reshape(1, n_feat, 1)

                for i in range(20):
                    next_pred = model.predict(curr_sequence)
                    predictions.append(next_pred[0, 0])

                    # Update the sequence: drop the first value and add the new prediction
                    curr_sequence = np.append(curr_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                col_name = f"{n_feat}_{batch_size}_{patience}"
                results_sheet[col_name] = predictions

                if 'actual_values' not in results_sheet.columns:
                    results_sheet['actual_values'] = checked_df['reading'].values[-20:]

                rmse = np.sqrt(mean_squared_error(checked_df['reading'].values[-20:], predictions))
                rmse_list.append(rmse)

                model.save(f"../Models/{n_feat}_{batch_size}_model.h5")

                if len(best_parameters) < 3:
                    best_parameters.append({'n_features': n_feat, 'dropout_rate': 0.2, 'batch_size': batch_size, 'patience': patience})
                    best_predictions.append(predictions)
                    best_models.append(model)

                else:
                    max_rmse_index = np.argmax(rmse_list[:3])
                    if rmse < rmse_list[:3][max_rmse_index]:
                        best_parameters[max_rmse_index] = {'n_features': n_feat, 'dropout_rate': 0.2, 'batch_size': batch_size, 'patience': patience}
                        best_predictions[max_rmse_index] = predictions
                        best_models[max_rmse_index] = model

    results_sheet.to_csv("../CSV_Files/results_sheet_Last_20_values.csv", index=False)

    # Save the best 3 parameters in the csv file
    best_parameters_df = pd.DataFrame(best_parameters)
    best_parameters_df.to_csv("../Models/best_parameters.csv", index=False)
    
    for i, model in enumerate(best_models):
        model.save(f"../Models/best_model_{i + 1}.h5")
    return best_parameters, best_predictions, best_models



def find_best_predictions_from_dataset():
    """
    This function will find the next 12 predictions using the best 3 models and train the model on the entire dataset using the best 3 parameters
    """

    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    df.set_index('Glucose_time', inplace=True)
    
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    
    # Drop all the columns which have unnamed in them
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # If reading_time is present in the dataset, then convert it to a proper date time format
    if 'Glucose_time' not in df.columns:
        df['reading_time'] = pd.to_datetime(df['reading_time'], unit='ms')
        df['reading_time'] = pd.to_datetime(df['reading_time'], format='%Y-%m-%d %H:%M:%S')
        df = df.rename(columns={'reading_time': 'Glucose_time'})
        df = df.sort_values(by='Glucose_time')

    # Set the Glucose_time to datetime format and set it as the index
    df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
    df.set_index('Glucose_time', inplace=True)

    checked_df = df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['reading'] = scaler.fit_transform(df[['reading']])

    time_series_data = df['reading'].values

    best_parameters = pd.read_csv("../Models/best_parameters.csv")

    # Now training the model on the entire dataset using the best 3 parameters
    best_predictions = []
    best_models = []
    
    final_predictions = pd.DataFrame()

    for i in range(3):
        n_feat = best_parameters['n_features'][i]
        batch_size = best_parameters['batch_size'][i]
        patience = best_parameters['patience'][i]

        print(f"Training model with n_features={n_feat}, dropout_rate=0.2, batch_size={batch_size}, patience={patience}")

        # Prepare data
        train_data_scaled = time_series_data
        test_data_scaled = time_series_data[-(n_feat + 12):]

        X_train, y_train = prepare_data(train_data_scaled, n_feat)
        X_test, y_test = prepare_data(test_data_scaled, n_feat)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define the model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_feat, 1)))
        if n_feat > 3:
            model.add(MaxPooling1D(pool_size=2))
        elif n_feat > 2:
            model.add(MaxPooling1D(pool_size=1))
        model.add(Dropout(0.2))

        model.add(LSTM(units=200, activation='tanh', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(units=150, activation='relu', return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(units=100, activation='relu', return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(units=50, activation='relu', return_sequences=False))  # Set return_sequences to False
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)

            # Train the model
        history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stop,lr_scheduler])

        predictions = []
        curr_sequence = X_test[0].reshape(1, n_feat, 1)

        for i in range(12):
            next_pred = model.predict(curr_sequence)
            predictions.append(next_pred[0, 0])

            # Update the sequence: drop the first value and add the new prediction
            curr_sequence = np.append(curr_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        col_name = f"{n_feat}_{batch_size}_{patience}"
        final_predictions[col_name] = predictions

        # There wont be the actual values in the final predictions as thery are the future predictions and cannot be determined in the present.

        model.save(f"../Models/{n_feat}_{batch_size}_model.h5")

    # Save the final predictions to a csv file
    final_predictions.to_csv("../CSV_Files/final_predictions_from_dataset.csv", index=False)

    return final_predictions







# In the given below case we will be given an array of some values and we need the model to predict the next 12 values of the array using the saved model.

def find_best_predictions_from_values(given_array):
    """
    This function will take the given array and predict the next 12 values using the best 3 models
    """

    model_1 = load_model("../Models/best_model_1.h5", compile=False)
    model_2 = load_model("../Models/best_model_2.h5", compile=False)
    model_3 = load_model("../Models/best_model_3.h5", compile=False)

    best_parameters = pd.read_csv("../Models/best_parameters.csv")
    p_model1 = best_parameters.iloc[0]
    p_model2 = best_parameters.iloc[1]
    p_model3 = best_parameters.iloc[2]

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    given_array = np.array(given_array)
    given_array = given_array.reshape(-1, 1)
    given_array = scaler.fit_transform(given_array)

    best_models = [model_1, model_2, model_3]
    best_parameters_list = [p_model1, p_model2, p_model3]

    model_1.compile(optimizer='adam', loss='mse')
    model_2.compile(optimizer='adam', loss='mse')
    model_3.compile(optimizer='adam', loss='mse')

    print("No problem till here")

    predictions_from_values = pd.DataFrame()

    for i in range(3):
        model = best_models[i]
        params = best_parameters_list[i]

        n_feat = int(params['n_features'])
        batch_size = int(params['batch_size'])
        patience = int(params['patience'])

        print(f"Processing model {i + 1} with n_features={n_feat}, batch_size={batch_size}, patience={patience}")

        if len(given_array) < n_feat:
            raise ValueError(f"Length of given_array ({len(given_array)}) is less than n_features ({n_feat})")

        test_data_scaled = scaler.transform(given_array).flatten()

        X_test, y_test = prepare_data(test_data_scaled, n_feat)

        if len(X_test) == 0:
            raise ValueError(f"Insufficient data to create test sequences with n_features={n_feat}")

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        predictions = []
        curr_sequence = X_test[0].reshape(1, n_feat, 1)

        for _ in range(12):
            next_pred = model.predict(curr_sequence)
            predictions.append(next_pred[0, 0])

            # Update the sequence: drop the first value and add the new prediction
            curr_sequence = np.append(curr_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        col_name = f"{n_feat}_{batch_size}_{patience}"
        predictions_from_values[col_name] = predictions

    # Save the final predictions to a csv file
    predictions_from_values.to_csv("../CSV_Files/predictions_from_values.csv", index=False)

    return predictions_from_values






def final_results():

    # Check whether the files already exist or not
    try:
        valid_final_preds = pd.read_csv("../CSV_Files/valid_predictions_lstm.csv")
        test_final_preds = pd.read_csv("../CSV_Files/test_predictions_lstm.csv")
        valid_check_preds = pd.read_csv("../CSV_Files/valid_check_predictions.csv")
        test_check_preds = pd.read_csv("../CSV_Files/test_check_predictions.csv")

        valid_rmse = valid_check_preds['RMSE'][0]
        valid_check = valid_check_preds[['Actual', 'Predictions']]

        test_rmse = test_check_preds['RMSE'][0]
        test_check = test_check_preds[['Actual', 'Predictions']]

        return valid_final_preds, test_final_preds, valid_check, test_check, valid_rmse, test_rmse

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


        # Testing the validation model and test model on some last values of the dataset
        valid_check_preds, test_check_preds, valid_rmse, test_rmse = find_df_test_validation_predictions(best_parameter_validation, best_parameter_test)

        # Adding a column named RMSE to the dataframes and storing the RMSE values
        valid_check_preds['RMSE'] = valid_rmse
        test_check_preds['RMSE'] = test_rmse

        # Save the dataframes to CSV files
        valid_check_preds.to_csv("../CSV_Files/valid_check_predictions.csv", index=False)
        test_check_preds.to_csv("../CSV_Files/test_check_predictions.csv", index=False)


        # Create a model on the entire dataset with the best validation dataset
        valid_final_preds, valid_model = lstm_on_entire_dataset(best_parameter_validation, 10)
        test_final_preds = lstm_on_entire_dataset(best_parameter_test, 10)

        # Saving the models into separate files for future use in a folder called Models
        valid_model.save("../Models/valid_model.h5")
        best_model_test.save("../Models/test_model.h5")

        # Save the valid_predictions to valid_predictions.csv and test_predictions to test_predictions.csv

        valid_final_preds.to_csv("../CSV_Files/valid_predictions_lstm.csv", index=False)
        test_final_preds[0].to_csv("../CSV_Files/test_predictions_lstm.csv", index=False)

    return valid_final_preds, test_final_preds[0], valid_check_preds, test_check_preds, valid_rmse, test_rmse


given_array = [325.22, 329.33, 333.44, 337.56, 341.67, 337.67, 336, 330, 324, 318, 312.5, 307, 298.33, 289.67, 281, 279, 277, 269, 266.75, 264.5, 262.25]
find_best_predictions_from_values(given_array)