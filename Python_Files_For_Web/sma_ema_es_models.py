import numpy as np
import pandas as pd



def Simple_Mov_Avg(evaluated_data):
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    evaluated_data['SMA'] = df['reading'].rolling(window=3).mean().shift(1)
    return evaluated_data


def Exp_Mov_Avg(evaluated_data):
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    evaluated_data['EMA'] = df['reading'].ewm(span=3, adjust=False, min_periods=0).mean().shift(1)
    return evaluated_data


def Exp_Smoothing(alpha, evaluated_data):
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    evaluated_data[f'ESA_{alpha}'] = df['reading'].ewm(alpha=alpha, adjust=False, min_periods=0).mean().shift(1)
    return evaluated_data


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean(((actual - predicted) ** 2)))


def predict_next_10_values_SMA():
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    lst = []
    lst.append(df['reading'].values[-3])
    lst.append(df['reading'].values[-2])
    lst.append(df['reading'].values[-1])
    
    for i in range(10):
        # Convert this list to a form on which we can apply the Rolling Mean
        series = pd.Series(lst)
        
        # Calculate the Rolling Mean
        rolling_mean = series.rolling(window=3).mean()

        # Append the Rolling Mean to the list
        lst.append(rolling_mean.values[-1])

    return lst[-10:]

def predict_next_10_values_EMA():
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
    # Get the last 3 readings
    last_readings = df['reading'].values[-3:].tolist()
    
    # Initialize the list with these readings
    ema_predictions = last_readings.copy()
    
    for _ in range(10):
        # Convert the list to a series
        series = pd.Series(ema_predictions[-3:])
        
        # Calculate the Exponential Moving Average
        ema = series.ewm(span=3, adjust=False, min_periods=0).mean().iloc[-1]
        
        # Append the EMA value to the list
        ema_predictions.append(ema)
    
    # Return only the last 10 predictions
    return ema_predictions[-10:]

def predict_next_10_values_ESA(alpha):
    df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")

    # Get the last 3 readings
    last_readings = df['reading'].values[-3:].tolist()
    
    # Initialize the list with these readings
    esa_predictions = last_readings.copy()
    
    for _ in range(10):
        # Convert the list to a series
        series = pd.Series(esa_predictions[-3:])
        
        # Calculate the Exponential Smoothing Average
        esa = series.ewm(alpha=alpha, adjust=False, min_periods=0).mean().iloc[-1]
        
        # Append the ESA value to the list
        esa_predictions.append(esa)
    
    # Return only the last 10 predictions
    return esa_predictions[-10:]