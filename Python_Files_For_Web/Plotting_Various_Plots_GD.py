import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_line_R_T(df, title, xlabel, ylabel):
    '''Plotting the line plot for the Glucose Reading vs Time'''
    plt.figsize(16, 10)
    sns.lineplot(x='Glucose_time', y='reading', data=df, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def plot_scatter_R_T(df, title, xlabel, ylabel):
    ''' Plotting the scatter plot for the Glucose Reading vs Time '''
    plt.figsize(16, 10)
    sns.scatterplot(x='Glucose_time', y='reading', data=df, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def plot_6_lag_plots(df):
    '''This function is for plotting the lag plots for the Glucose Reading data. The lags are 1, 6, 12, 36, 72, 144.'''
    fig, axs = plt.subplots(2, 3, figsize=(16, 16))

    pd.plotting.lag_plot(df['reading'], lag=1, ax=axs[0, 0])
    axs[0, 0].set_title('Lag 1')

    pd.plotting.lag_plot(df['reading'], lag=12, ax=axs[0, 2])
    axs[0, 2].set_title('Lag 12')

    pd.plotting.lag_plot(df['reading'], lag=36, ax=axs[1, 0])
    axs[1, 0].set_title('Lag 36')

    pd.plotting.lag_plot(df['reading'], lag=72, ax=axs[1, 1])
    axs[1, 1].set_title('Lag 72')

    pd.plotting.lag_plot(df['reading'], lag=6, ax=axs[0, 1])
    axs[0, 1].set_title('Lag 6')

    pd.plotting.lag_plot(df['reading'], lag=144, ax=axs[1, 2])
    axs[1, 2].set_title('Lag 144')

    plt.show()


def plot_acf(df, xlabel, ylabel):
    '''Plotting the Auto Correlation Plot for the Glucose Reading data.'''
    plt.figure(figsize=(16, 8))
    plot_acf(df['reading'], lags=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Auto Correlation Plot of Glucose Reading")

    plt.show()


def plot_pacf(df, xlabel, ylabel):
    plt.figure(figsize=(16, 8))
    plot_pacf(df['reading'], lags=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Partial Auto Correlation Plot of Glucose Reading")

    plt.show()


def plot_hist_with_kde(df, xlabel, ylabel):
    '''Plotting the Histogram with KDE for the Glucose Reading data.'''
    plt.figure(figsize=(16, 8))
    sns.histplot(df['reading'], kde=True, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Histogram with KDE of Glucose Reading")

    plt.show()


def compare_original_resampled(df, resampled_df):
    '''Comparing the original data with the resampled data.'''
    plt.figure(figsize=(16, 8))
    sns.lineplot(x='Glucose_time', y='reading', data=df, color='blue', label='Original Data')
    sns.lineplot(x='Glucose_time', y='reading', data=resampled_df, color='red', label='Resampled Data')
    plt.xlabel('Time')
    plt.ylabel('Glucose Reading')
    plt.title('Original Data vs Resampled Data')
    plt.legend()

    plt.show()


# The function can have a minimum of 3 and a maximum of 5 parameters, which can be chosen by the user. The first parameter is the original data, the second parameter is the Simple Moving Average (SMA), the third parameter is the Exponential Moving Average (EMA), and the fourth parameter is the Exponential Smoothing Average With alpha value of x (ESA) [optional]. The fifth parameter is is the Exponential Smoothing Average With alpha value of y (ESA) [optional]

# To mark a parameter as optional, we can use args and kwargs. The args will be the optional parameters, and the required parameters will be passed as the normal parameters. The kwargs will be the optional parameters that are passed as key-value pairs.

def plot_moving_averages(original, sma, ema, *args, **kwargs):
    '''Plotting the Moving Averages for the Glucose Reading data.'''
    plt.figure(figsize=(16, 8))
    sns.lineplot(x='Glucose_time', y='reading', data=original, color='blue', label='Original Data')
    sns.lineplot(x='Glucose_time', y='reading', data=sma, color='red', label='Simple Moving Average')
    sns.lineplot(x='Glucose_time', y='reading', data=ema, color='green', label='Exponential Moving Average')

    for key, value in kwargs.items():
        sns.lineplot(x='Glucose_time', y='reading', data=value, label=f'Exponential Smoothing Average with alpha value of {key}')

    for arg in args:
        sns.lineplot(x='Glucose_time', y='reading', data=arg, label=f'Exponential Smoothing Average with alpha value of {arg}')

    plt.xlabel('Time')
    plt.ylabel('Glucose Reading')
    plt.title('Comparison of Moving Averages with Original Data')
    plt.legend()

    plt.show()


