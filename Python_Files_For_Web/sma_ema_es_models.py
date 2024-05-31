import numpy as np

def Simple_Mov_Avg(data, window_size):
    sma_data = data.rolling(window=window_size).mean().shift(1)
    return sma_data


def Exp_Mov_Avg(data, window_size):
    ema_data = data.ewm(span=window_size, adjust=False).mean().shift(1)
    return ema_data


def Exp_Smoothing(data, alpha):
    exp_data = data.ewm(alpha=alpha, adjust=False, min_periods=0).mean().shift(1)
    return exp_data


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean(((actual - predicted) ** 2)))