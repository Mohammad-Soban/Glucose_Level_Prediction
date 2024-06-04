import streamlit as st
import pandas as pd
import numpy as np
import warnings

import sys

warnings.filterwarnings("ignore")
sys.path.append("../Python_Files_For_Web")

from Data_Cleaning_From_CSV import cleaning_csv_file
from Plotting_Various_Plots_GD import plot_line_R_T, plot_scatter_R_T,  plot_6_lag_plots, plot_acf_df, plot_pacf_df, plot_hist_with_kde, compare_original_resampled, plot_moving_averages
from sma_ema_es_models import Simple_Mov_Avg, Exp_Mov_Avg, Exp_Smoothing, calculate_rmse, predict_next_10_values_SMA, predict_next_10_values_EMA, predict_next_10_values_ESA
from implement_arima import implement_arima_df, get_arima_rmse
from implement_LSTM_Model import Find_Best_Params_On_Validation_data

# Add a title as centered text
st.title("Glucose Level Predcition")

# Add a button to upload a CSV File
glucose_data = st.file_uploader("Upload a CSV file", type=["csv"])

# Open this file using pandas and save it in a new csv file named glucose_data.csv
if glucose_data is not None:
    data = pd.read_csv(glucose_data)
    data.to_csv("../CSV_Files/glucose_data.csv", index=False)
    data['Glucose_time'] = pd.to_datetime(data['Glucose_time'])

    # Clean the csv file and get a resampled version of the data
    df = cleaning_csv_file()


# If the file is uploaded then display the options sidebar to the user.
if glucose_data is not None:
    # Add a sidebar with the following options
    st.sidebar.title("Options")

    Options = ["View Original Data", "View Resampled Data", "View Prediction Data Using Simple Models", "View Predictions using ARIMA", "View Predictions Using LSTM", "View Plots of Resampled vs Original Data", "View Plots of Resampled Data", "View Plots of Prediction Data", "View RMSE Values"]

    # Display the options in a list format
    option = st.sidebar.selectbox("Select an option", Options)

    # If the user selects the option "View Original Data" then display the original data
    if option == "View Original Data":
        st.title("Original Data")
        st.write(data)

    
    # If the user selects the option "View Resampled Data" then display the resampled data
    elif option == "View Resampled Data":
        st.title("Resampled Data")
        st.write(df)

    
    elif option == 'View Plots of Resampled vs Original Data':
        st.title("Plots of Resampled vs Original Data")
        compare_original_resampled(data, df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

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

    elif option == "View Prediction Data Using Simple Models":
        st.title("Prediction Data Using Simple Models")
        # Call all the models and get their predictions
        evaluated_data = Simple_Mov_Avg()
        evaluated_data = Exp_Mov_Avg()
        evaluated_data = Exp_Smoothing(0.3)
        evaluated_data = Exp_Smoothing(0.5)
        evaluated_data = Exp_Smoothing(0.7)

        # Display the predictions
        st.write(evaluated_data)

        # Also I want to predict the next 10 values using every model and append it in a new dataframe
        # The new dataframe will have columns Glucose_time, Prediction using SMA, Prediction Using EMA, Prediction using ESA0.3, Prediction using ESA0.5, Prediction using ESA0.7.
        # Note that the predictions will be for the next 10 values after the last value in the original data.
        # Display this dataframe as well.

        st.title("Next 10 Predictions using Simple Models")
        next_10_predictions = pd.DataFrame()
        next_10_predictions['Glucose_time'] = pd.date_range(start=evaluated_data['Glucose_time'].iloc[-1], periods=10, freq='5min')
        
        next_10_predictions['Prediction using SMA'] = predict_next_10_values_SMA()
        next_10_predictions['Prediction using EMA'] = predict_next_10_values_EMA()
        next_10_predictions['Prediction using ESA0.3'] = predict_next_10_values_ESA(0.3)
        next_10_predictions['Prediction using ESA0.5'] = predict_next_10_values_ESA(0.5)
        next_10_predictions['Prediction using ESA0.7'] = predict_next_10_values_ESA(0.7)

        st.dataframe(next_10_predictions)

    elif option == "View Predictions using ARIMA":
        st.title("Predictions using ARIMA")
        st.write("Enter the number of future values you want to predict")

        period = st.number_input("Enter the number of future values you want to predict", min_value=1, max_value=21)
        
        # Get the predictions using ARIMA
        predictions = implement_arima_df(period)

        # Display the predictions
        st.write(predictions)

    elif option == "View Predictions using LSTM":
        st.title("Predictions using LSTM")
        st.write("Finding the best parameters for the LSTM model")
        Best_Paramas, Best_Model, Best_Predictions= Find_Best_Params_On_Validation_data()
        
        

        