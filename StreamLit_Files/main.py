import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys

warnings.filterwarnings("ignore")
sys.path.append("../Python_Files_For_Web")

from tensorflow.keras.models import load_model
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from Data_Cleaning_From_CSV import cleaning_csv_file
from Plotting_Various_Plots_GD import plot_line_R_T, plot_scatter_R_T,  plot_6_lag_plots, plot_acf_df, plot_pacf_df, plot_hist_with_kde, compare_original_resampled, plot_moving_averages
from sma_ema_es_models import Simple_Mov_Avg, Exp_Mov_Avg, Exp_Smoothing, calculate_rmse, predict_next_10_values_SMA, predict_next_10_values_EMA, predict_next_10_values_ESA
from implement_arima import implement_arima_df, get_arima_rmse, get_test_preds
from implement_LSTM_Model import final_results, find_df_test_validation_predictions
from User_input_Predictions import preds_from_user_input

import time

data = None

def is_70_percent_same(df1, df2):
    # Ensure the 'readings' column is present in both dataframes
    if 'reading' not in df1.columns or 'reading' not in df2.columns:
        raise ValueError("Both dataframes must contain a 'reading' column")

    # Extract the 'readings' columns
    reading1 = df1['reading']
    reading2 = df2['reading']
    
    # Align the data, filling missing values with NaN
    reading1, reading2 = reading1.align(reading2, fill_value=np.nan)
    
    # Compare the values in the 'reading' columns
    matches = reading1 == reading2
    
    # Calculate the percentage of matching values
    total_values = matches.size
    matching_values = matches.sum()
    match_percentage = (matching_values / total_values) * 100
    
    # Return True if the match percentage is 70% or higher, otherwise False
    return match_percentage >= 70

# Add a title as centered text
st.title("Glucose Level Predcition")


# Add a button to upload a CSV File
glucose_data = st.file_uploader("Upload a CSV file", type=["csv"])


if glucose_data is not None:
    data = pd.read_csv(glucose_data)

try:
    ck1 = pd.read_csv("../CSV_Files/glucose_data.csv")

except Exception as e:
    pass


# Open this file using pandas and save it in a new csv file named glucose_data.csv
if data is not None:
    data.to_csv("../CSV_Files/glucose_data.csv", index=False)
    data['Glucose_time'] = pd.to_datetime(data['Glucose_time'])
    # Clean the csv file and get a resampled version of the data
    df = cleaning_csv_file()


# If the file is uploaded then display the options sidebar to the user.
if glucose_data is not None:
    # Add a sidebar with the following options
    st.sidebar.title("Options")

    Options = ["View Original Data", "View Resampled Data", "View Prediction Data Using Simple Models", "View Predictions using ARIMA", "View Predictions Using LSTM", "Get Predictions by Giving Data", "View Plots of Resampled vs Original Data", "View Plots of Resampled Data", "View Plots of Prediction Data", ]

    # Display the options in a list format
    option = st.sidebar.selectbox("Select an option", Options)

    # Option 1
    # If the user selects the option "View Original Data" then display the original data
    if option == "View Original Data":
        st.title("Original Data")
        st.write(data)


    # Option 2    
    # If the user selects the option "View Resampled Data" then display the resampled data
    elif option == "View Resampled Data":
        st.title("Resampled Data")
        st.write(df)

    
    # Option 3
    # If the user selects the option "View Plots of Resampled vs Original Data" then display the plots of the resampled data and the original data
    elif option == 'View Plots of Resampled vs Original Data':
        st.title("Plots of Resampled vs Original Data")
        compare_original_resampled(data, df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    # Option 4
    # If the user selects the option "View Plots of Resampled Data" then display the plots of the resampled data
    elif option == "View Plots of Resampled Data":
        # Give Options to the user to select the type of plot they want to see
        st.title("Plots of Resampled Data")
        st.write("Select the type of plot you want to see")
        plot_options = ["Line Plot", "Scatter Plot", "Lag Plots", "ACF Plot", "PACF Plot", "Histogram with KDE",]

        plot_option = st.selectbox("Select the type of plot", plot_options)

        if plot_option == "Line Plot":
            st.title("Line Plot of Resampled Data")
            plot_line_R_T(df, "Line Plot", "Time", "Glucose Reading")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        elif plot_option == "Scatter Plot":
            st.title("Scatter Plot of Resampled Data")
            plot_scatter_R_T(df, "Scatter Plot", "Time", "Glucose Reading")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        elif plot_option == "Lag Plots":
            st.title("Lag Plots of Resampled Data")
            plot_6_lag_plots(df)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        elif plot_option == "Histogram with KDE":
            st.title("Histogram with the KDE of Resampled Data")
            plot_hist_with_kde(df, "Histogram with KDE", "Glucose Reading")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        elif plot_option == "ACF Plot":
            st.title("ACF Plot of Resampled Data")
            plot_acf_df(df, "Lags", "Glucose Reading")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        elif plot_option == "PACF Plot":
            st.title("PACF Plot of Resampled Data")
            plot_pacf_df(df, "Lags", "Glucose Reading")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()


    # Option 5
    # If the user selects the option "View Prediction Data Using Simple Models" then display the predictions using Simple Models such as SMA, EMA and ESA.
    elif option == "View Prediction Data Using Simple Models":
        st.title("Prediction Data Using Simple Models")
        # Call all the models and get their predictions
        df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
        evaluated_data = df.copy()

        evaluated_data = Simple_Mov_Avg(evaluated_data)
        evaluated_data = Exp_Mov_Avg(evaluated_data)
        evaluated_data = Exp_Smoothing(0.3, evaluated_data)
        evaluated_data = Exp_Smoothing(0.5, evaluated_data)
        evaluated_data = Exp_Smoothing(0.7, evaluated_data)
        
        # Save the evaluated data to a csv file
        # evaluated_data.to_csv("../CSV_Files/moving_averages_predictions.csv", index=False)

        # Display the test predictions that is the last 10 values of the predictions
        st.write("Test Predictions using Simple Models")
        test_df = evaluated_data.iloc[-10:]
        st.write(test_df)


        # st.write(evaluated_data)

        st.write("Next 10 Predictions using Simple Models")
        next_10_predictions = pd.DataFrame()
        next_10_predictions['Glucose_time'] = pd.date_range(start=evaluated_data['Glucose_time'].iloc[-1], periods=10, freq='5min')
        
        next_10_predictions['Prediction using SMA'] = predict_next_10_values_SMA()
        next_10_predictions['Prediction using EMA'] = predict_next_10_values_EMA()
        next_10_predictions['Prediction using ESA0.3'] = predict_next_10_values_ESA(0.3)
        next_10_predictions['Prediction using ESA0.5'] = predict_next_10_values_ESA(0.5)
        next_10_predictions['Prediction using ESA0.7'] = predict_next_10_values_ESA(0.7)

        st.dataframe(next_10_predictions)

        # If any value of the predictions is > 126 then write "Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin.
        if next_10_predictions['Prediction using SMA'].max() > 126 or next_10_predictions['Prediction using EMA'].max() > 126 or next_10_predictions['Prediction using ESA0.3'].max() > 126 or next_10_predictions['Prediction using ESA0.5'].max() > 126 or next_10_predictions['Prediction using ESA0.7'].max() > 126:
            st.write("Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin.")

        
        # If any value of the predictions is < 60 then write "Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar."
        if next_10_predictions['Prediction using SMA'].min() < 60 or next_10_predictions['Prediction using EMA'].min() < 60 or next_10_predictions['Prediction using ESA0.3'].min() < 60 or next_10_predictions['Prediction using ESA0.5'].min() < 60 or next_10_predictions['Prediction using ESA0.7'].min() < 60:
            st.write("Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar.")

        
        # Case for normal glucose levels
        if next_10_predictions['Prediction using SMA'].max() <= 126 and next_10_predictions['Prediction using EMA'].max() <= 126 and next_10_predictions['Prediction using ESA0.3'].max() <= 126 and next_10_predictions['Prediction using ESA0.5'].max() <= 126 and next_10_predictions['Prediction using ESA0.7'].max() <= 126 and next_10_predictions['Prediction using SMA'].min() >= 60 and next_10_predictions['Prediction using EMA'].min() >= 60 and next_10_predictions['Prediction using ESA0.3'].min() >= 60 and next_10_predictions['Prediction using ESA0.5'].min() >= 60 and next_10_predictions['Prediction using ESA0.7'].min() >= 60:
            st.write("Your Glucose Levels are normal for the next 1 hour. Keep up the good work!")   

        # Save this to a csv file
        next_10_predictions.to_csv("../CSV_Files/moving_averages_predictions.csv", index=False)


    # Option 6
    # If the user selects the option "View Predictions using ARIMA" then display the predictions using ARIMA
    elif option == "View Predictions using ARIMA":
        st.title("Predictions using ARIMA")

        period = st.number_input("Enter the number of future values you want to predict", value=10, min_value=1, max_value=21, step=1)

        test_predictions = get_test_preds(period)
        
        st.write("Test Predictions using ARIMA")
        random = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
        test_predictions_df = random[-period:]
        test_predictions_df['Preds'] = test_predictions

        # print(test_predictions_df)

        st.dataframe(test_predictions_df)
        
        # Get the predictions using ARIMA
        predictions = implement_arima_df(period)

        # save the predictions to a csv file
        predictions.to_csv("../CSV_Files/arima_predictions.csv", index=False)

        # Display the predictions
        
        st.write(predictions)

        # If the predictions are greater than 126 then write "Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin."
        if predictions['reading'].max() > 126:
            st.write("Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin in case it goes up.")

        # If the predictions are less than 60 then write "Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar."
        if predictions['reading'].min() < 60:
            st.write("Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar.")

        # Case for normal glucose levels
        if predictions['reading'].max() <= 126 and predictions['reading'].min() >= 60:
            st.write("Your Glucose Levels are normal for the next 1 hour. Keep up the good work!")


    # Option 7
    # If the user selects the option "View Predictions Using LSTM" then display the predictions using LSTM
    elif option == "View Predictions Using LSTM":
        st.title("Predictions using LSTM")
        
        # Check if the data in ck1 is same as the data which is uploaded or not
        if (ck1 == data).all().all():            
            # We will print the predictions from the csv_files
            valid_final_preds, test_final_preds, valid_check_preds, test_check_preds, valid_rmse, test_rmse = final_results()

            # Rename the columns Prediction_Valid and Prediction_Test to Prediction
            valid_final_preds.rename(columns={'Prediction_Valid': 'Prediction'}, inplace=True)
            test_final_preds.rename(columns={'Prediction_Test': 'Prediction'}, inplace=True)

            st.write("Last Some Values Predicted By Validation Model")
            st.write(valid_check_preds)

            st.write("Validation Data Predictions")
            st.write(valid_final_preds)

            st.write("Last Some Values Predicted By Test Model")
            st.write(test_check_preds)

            st.write("Test Data Predictions")
            st.write(test_final_preds)
            
            # Write this with big font size
            st.write("Getting the RMSE Values Ready For You")
            st.write("RMSE Value for Validation Data:", valid_rmse)
            st.write("RMSE Value for Test Data:", test_rmse)  


        elif is_70_percent_same(ck1, data):
            
            test_model = load_model('../Models/test_model.h5', custom_objects={'mse': mean_squared_error})
            valid_model = load_model('../Models/valid_model.h5', custom_objects={'mse': mean_squared_error})

            df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
            df['Glucose_time'] = pd.to_datetime(df['Glucose_time'])
            df.set_index('Glucose_time', inplace=True)

            values = df['reading'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(values)

            best_params_valid = pd.read_csv("../Models/best_parameters_lstm.csv")
            best_params_test = pd.read_csv("../Models/best_parameters_test_lstm.csv")

            # n_features,patience

            n_feat_valid = best_params_valid['n_features'].values[0]
            n_feat_test = best_params_test['n_features'].values[0]

            patience_valid = best_params_valid['patience'].values[0]
            patience_test = best_params_test['patience'].values[0]

            # Get the predictions for the last 10 values of the validation and test data
            df_validation, df_test, valid_rmse, test_rmse = find_df_test_validation_predictions(n_feat_valid, n_feat_test, patience_valid, patience_test, valid_model, test_model, scaler, scaled_values)

            st.write("Validation Data Predictions For Last Some Values of Dataset")
            st.write(df_validation)

            st.write("Test Data Predictions For Last Some Values of Dataset")
            st.write(df_test)


            input_data = scaled_values[-n_feat_valid:].reshape((1, n_feat_valid, 1))
            predictions_valid = []
            for _ in range(10):
                next_value = valid_model.predict(input_data)
                predictions_valid.append(next_value[0, 0])
                input_data = np.append(input_data[:, 1:, :], next_value).reshape((1, n_feat_valid, 1))

            predictions_valid = scaler.inverse_transform(np.array(predictions_valid).reshape(-1, 1))
            print(predictions_valid)

            input_data = scaled_values[-n_feat_test:].reshape((1, n_feat_test, 1))
            predictions_test = []
            for _ in range(10):
                next_value = test_model.predict(input_data)
                predictions_test.append(next_value[0, 0])
                input_data = np.append(input_data[:, 1:, :], next_value).reshape((1, n_feat_test, 1))
            
            predictions_test = scaler.inverse_transform(np.array(predictions_test).reshape(-1, 1))
            print(predictions_test)

            # Create a new Data Frame with the Future 10 values of Glucose Timeings and both the predictions appended to it as columns
            next_10_predictions = pd.DataFrame()
            next_10_predictions['Glucose_time'] = pd.date_range(start=df.index[-1], periods=10, freq='5min')
            next_10_predictions['Prediction_Valid'] = predictions_valid
            next_10_predictions['Prediction_Test'] = predictions_test

            # Save the predictions separately for validation and test data
            valid_final_preds = next_10_predictions[['Glucose_time', 'Prediction_Valid']]
            test_final_preds = next_10_predictions[['Glucose_time', 'Prediction_Test']]

            valid_final_preds.to_csv("../CSV_Files/valid_predictions_lstm.csv", index=False)
            test_final_preds.to_csv("../CSV_Files/test_predictions_lstm.csv", index=False)

            # Rename the columns Prediction_Valid and Prediction_Test to Prediction
            valid_final_preds.rename(columns={'Prediction_Valid': 'Prediction'}, inplace=True)
            test_final_preds.rename(columns={'Prediction_Test': 'Prediction'}, inplace=True)

            st.write("Validation Data Predictions")
            st.write(valid_final_preds)

            st.write("Test Data Predictions")
            st.write(test_final_preds)

            # Write this with big font size
            st.write("Getting the RMSE Values Ready For You")
            st.write("RMSE Value for Validation Data:", valid_rmse)
            st.write("RMSE Value for Test Data:", test_rmse)    
        
        # Else there are 2 conditions 1) There is no data in ck1 and 2) The data is completely different from the data in ck1.
        else:
            try:
                # Delete the 2 files test_predictions_lstm.csv and valid_predictions_lstm.csv from the ../CSV_Files folder
                os.remove("../CSV_Files/test_predictions_lstm.csv")
                os.remove("../CSV_Files/valid_predictions_lstm.csv")

            except Exception as e:
                pass

            finally:
                valid_final_preds, test_final_preds, valid_check_preds, test_check_preds, valid_rmse, test_rmse = final_results()

                st.write("Last Some Values Predicted By Validation Model")
                st.write(valid_check_preds)

                st.write("Validation Data Predictions")
                st.write(valid_final_preds)

                st.write("Last Some Values Predicted By Test Model")
                st.write(test_check_preds)

                st.write("Test Data Predictions")
                st.write(test_final_preds)

                # Write this with big font size
                st.write("Getting the RMSE Values Ready For You")
                st.write("RMSE Value for Validation Data:", valid_rmse)
                st.write("RMSE Value for Test Data:", test_rmse)


        # If the predictions are greater than 126 then write "Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin."
        if (valid_final_preds['Prediction'].max() + test_final_preds['Prediction'].max())/2 > 126:
            st.write("Your Glucose Level can go high in the next 1 hour. Please visit a doctor to have insulin in case it goes up.")

        # If the predictions are less than 60 then write "Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar."
        if (valid_final_preds['Prediction'].min() + test_final_preds['Prediction'].min())/2 < 60:
            st.write("Your Glucose Level can go low in the next 1 hour. Please constantly monitor your glucose levels and if it goes lower than 60 then have some sugar.")

        # Case for normal glucose levels
        if (valid_final_preds['Prediction'].max() + test_final_preds['Prediction'].max())/2 <= 126 and (valid_final_preds['Prediction'].min() + test_final_preds['Prediction'].min())/2 >= 60:
            st.write("Your Glucose Levels are normal for the next 1 hour. Keep up the good work!")


    elif option == "Get Predictions by Giving Data":
        num_values = st.number_input("Enter the number of values you have", min_value=1, max_value=20, step=1)
        
        values = st.text_input("Enter the values of the glucose readings separated by commas")


        try:
            values = values.split(',')
            values = [float(value) for value in values]

            if len(values) == num_values:
                
                final_df, final_preds = preds_from_user_input(num_values, values)
                st.write("The predictions if this model was used to predict the last some values of the dataset.")
                st.write(final_df)


                st.write("If the above mentioned values are readings of glucose levels then the predictions for the next 10 values are as follows:")
                st.write(final_preds)

            else:
                raise ValueError(f"{len(values)} values entered. Please enter {num_values} values")


        except Exception as e:
            # Hide the Traceback error
            text = e.with_traceback(None)
            st.write(text)

    elif option == 'View Plots of Prediction Data': 
        try:
            st.title("Plots of Prediction Data")

            ma_df = pd.read_csv("../CSV_Files/moving_averages_predictions.csv")
            arima_df = pd.read_csv("../CSV_Files/arima_predictions.csv")
            validation_lstm_df = pd.read_csv("../CSV_Files/valid_predictions_lstm.csv")
            test_lstm_df = pd.read_csv("../CSV_Files/test_predictions_lstm.csv")

            plt.figure(figsize=(16, 8))
            # sns.lineplot(x='Glucose_time', y='reading', data=df, color='blue', label='Original Data')
            sns.lineplot(x='Glucose_time', y='Prediction using SMA', data=ma_df, color='red', label='Simple Moving Average')
            sns.lineplot(x='Glucose_time', y='Prediction using EMA', data=ma_df, color='green', label='Exponential Moving Average')
            sns.lineplot(x='Glucose_time', y='Prediction using ESA0.3', data=ma_df, color='purple', label='Exponential Smoothing Average with alpha 0.3')
            sns.lineplot(x='Glucose_time', y='Prediction using ESA0.5', data=ma_df, color='yellow', label='Exponential Smoothing Average with alpha 0.5')
            sns.lineplot(x='Glucose_time', y='Prediction using ESA0.7', data=ma_df, color='orange', label='Exponential Smoothing Average with alpha 0.7')
            sns.lineplot(x='Glucose_time', y='reading', data=arima_df, color='black', label='ARIMA Predictions')
            sns.lineplot(x='Glucose_time', y='Prediction', data=validation_lstm_df, color='pink', label='Validation LSTM Predictions')
            sns.lineplot(x='Glucose_time', y='Prediction', data=test_lstm_df, color='brown', label='Test LSTM Predictions')

            plt.xlabel('Time')
            plt.ylabel('Glucose Reading')
            plt.title('Predictions using Various Models')

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            

        except Exception as e:
            st.title("Please first view all the kinds of predictions to view the plots.")
            st.write("Error:", e)