import pandas as pd
import numpy as np

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# Function to make the data compatible with ARIMA
# print(data.columns)

def implement_arima_df(period, start_p=0, start_q=0, max_p=5, max_q=5):
  data = pd.read_csv("../CSV_Files/glucose_data_resampled.csv")
  model = auto_arima(data['reading'], start_p=start_p, start_q=start_q, max_p=max_p, max_q=max_q, seasonal=False, stepwise=True)
  predictions = model.predict(n_periods=period)

  # Create next_values DataFrame with timestamps
  next_values = pd.DataFrame(index=pd.date_range(start=data['Glucose_time'].iloc[-1], periods=period+1, freq='5min')[1:], columns=['Glucose_time', 'reading'])
  next_values['Glucose_time'] = next_values.index

  # Assign predictions to 'reading' column
  next_values['reading'] = predictions.values
  next_values.reset_index(drop=True, inplace=True)

  return next_values


def get_arima_rmse(test_data, predictions):
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    return rmse