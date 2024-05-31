from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

def make_arima_data(upto_data, data):
    train_data = data[:upto_data]
    test_data = data[upto_data:]

    return train_data, test_data


def implement_arima(train_data, period, start_p=0, start_q=0, max_p=5, max_q=5):
    
    model = auto_arima(train_data, start_p=start_p, start_q=start_q, max_p=max_p, max_q=max_q, seasonal=False, stepwise=True)

    predictions = model.predict(n_periods=period)

    return predictions


def get_arima_mse(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    return mse