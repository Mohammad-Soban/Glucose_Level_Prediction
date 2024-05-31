from statsmodels.tsa.stattools import kpss, adfuller


def adftest(data):
    adftest = adfuller(data)
    p_value = adftest[1]
    if p_value < 0.05:
        print("With ADF Test, we conclude that the data is stationary")

    else:
        print("With ADF Test, we conclude that the data is not stationary")


def kpsstest(data):
    kpss_test = kpss(data)
    p_value = kpss_test[1]
    if p_value < 0.05:
        print("With KPSS Test, we conclude that the data is not stationary")

    else:
        print("With KPSS Test, we conclude that the data is stationary")