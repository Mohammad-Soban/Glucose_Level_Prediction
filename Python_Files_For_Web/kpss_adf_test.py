from statsmodels.tsa.stattools import kpss, adfuller
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")


def adftest():
    data = df['reading']

    adftest = adfuller(data)
    p_value = adftest[1]
    if p_value < 0.05:
        print("With ADF Test, we conclude that the data is stationary")

    else:
        print("With ADF Test, we conclude that the data is not stationary")


def kpsstest():

    data = df['reading']

    kpss_test = kpss(data)
    p_value = kpss_test[1]
    if p_value < 0.05:
        print("With KPSS Test, we conclude that the data is not stationary")

    else:
        print("With KPSS Test, we conclude that the data is stationary")

adftest()

kpsstest()