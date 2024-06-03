import numpy as np
import pandas as pd


df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
evaluated_data = df.copy()

def Simple_Mov_Avg():
    sma_data = df['reading'].rolling(window=3).mean().shift(1)
    evaluated_data['SMA'] = sma_data
    return evaluated_data


def Exp_Mov_Avg():
    ema_data = df['reading'].ewm(span=3, adjust=False, min_periods=0).mean().shift(1)
    evaluated_data['EMA'] = ema_data
    return evaluated_data


def Exp_Smoothing(alpha):
    exp_data = df['reading'].ewm(alpha=alpha, adjust=False, min_periods=0).mean().shift(1)
    evaluated_data[f'ESA{alpha}'] = exp_data
    return evaluated_data


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean(((actual - predicted) ** 2)))


