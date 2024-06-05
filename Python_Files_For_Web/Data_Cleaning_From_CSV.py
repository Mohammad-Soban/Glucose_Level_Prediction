import pandas as pd
import numpy as np



def cleaning_csv_file():
    file_path = "../CSV_Files/glucose_data.csv"
    data = pd.read_csv(file_path)
    
    # If there are any columns with the column name containing unnamed then drop them.
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Convert the glucose_time column to datetime format
    data['Glucose_time'] = pd.to_datetime(data['Glucose_time'], format='%Y-%m-%d %H:%M:%S')

    # Take the glucose_time and reading columns in df and resample that to 5 minutes of interval using the interpolation technique with the method as linear
    df = data[['Glucose_time', 'reading']]
    df = df.set_index('Glucose_time')
    df = df.resample('5min').mean().interpolate(method='linear')

    # Reset the index of the df and save it to a new csv file

    # The file path is in the format "C:/Users/BMVSI-138/Desktop/Glucose_Prediction/glucose_data.csv"
    # Save the file to the same location with the name filename_resampled.csv
    # Save with the index as well
    file_path_resampled = file_path.split("/")
    file_path_resampled[-1] = file_path_resampled[-1].replace(".csv", "_resampled.csv")
    file_path_resampled = "/".join(file_path_resampled)
    
    # Saving the dataframe to the new csv file
    df.to_csv(file_path_resampled, index=True)
    return df