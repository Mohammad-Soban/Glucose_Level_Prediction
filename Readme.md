# <center>  Time Series Modelling for Glucose Level Predictions </center>

Welcome to the Glucose Level Prediction! This project demonstrates the use of a various kinds of model which include Moving Avregase Model, Auto Regressive Model, ARMA/ARIMA Model and Deep Learning Model in the form of LSTM which is a type of RNN Model to predict the future values of glucose readings from the dataset using various libraries of Python

## Overview 📄

This project is aimed at building the above mentioned models for Glucose Level Prediction. The project also involves resampling the data and providing visualizations and adding it to the streamlit application.

## Dataset 📚

The given dataset consists of 250 time stamps along with their corresponding Readings.

## Data Preprocessing

The given data does not have any outliers, null values or duplicates, so we can skip the three steps. But for making it a time series data, the data needs to evenly spaced in terms of the time. So we resample the data using the linear interpolation, and we will use the mean in case of any challenges we face in linear interpolation.

Also the Glucose_time column is in the object data type. So we convert that column in Date-Time Object. Also we need to remove all the columns which are useless as for example the "Unnamed : 0" column and the Glucose_time column which is the same as the  Reading_time Column

## Model Training 🏋️‍♂️

#### Moving Averages Model

Moving Average (MA) is a statistical technique used in time series analysis to estimate the underlying trend or pattern in the data by averaging the values of a certain number of preceding periods along with their error terms.
<br>
The Moving Averages Model can be subdivided in 3 Models
<ul>
  <li>Simple Moving Average</li>
  <li>Exponential Moving Average</li>
  <li>Exponential Smoothening</li>
</ul>

The Moving Averages Model works on a sliding window concept, such that it will consider all the elements inside the window and make the next prediction on the basis of it.

Pandas have inbuilt functions to implement the SMA, EMA, ESA Models and using these models we need to predict the next values.


#### ARMA/ARIMA Model

AR stands for Auto Regressive, MA stands for Moving Averages and I stands for Integrated. The above mentioned models are combinations of these models. The ARMA Model has 2 parameters the q value and the p value. The q value representes the number of lagged forecast errors while the p value represents the number of lagged observations, while the d value is additional in ARIMA Model which tells number of time observations need to be differentiated.

Steps : <br>
<li> Determine whether to use ARMA Model or ARIMA Model: It can be determined by testing for stationarity of the data using the KPSS or ADF Test.</li>

<li> ARMA/ARIMA model can be directly imported from pmdarima library.</li>

<li> Find the parameters for the ARMA Model (p, q and d(if required)).</li>

<li> Find the predictions and calculate the RMSE value.</li>


#### LSTM MODEL
The problem with previous models is that it loses the information that appeared long back and only retains the information coming in the recent times. The activation functions are constantly updated and as a result they quickly lose the memory. This probelem is called the <b> Vanishing Gradient Problem </b>

<br>

This problem is solved by LSTM Model which has the functionality to retain the important information and forget the information which is not important.

<br>

The LSTM Model has the following components:
<ul>
  <li> Forget Gate: It decides what information to throw away and what information to keep.</li>
  <li> Input Gate: It decides which values to update.</li>
  <li> Output Gate: It decides what the next hidden state should be.</li>
  <li> Memory Cell State : It is responsible to capture and retain long-term dependencies in sequential data </li>
  <li> Hidden State: : It contains the information that is passed to the next time step.</li>
</ul>

The LSTM Model is implemented using the TensorFlow and Keras Library. The LSTM Model is trained on the dataset and the predictions are made.

The Steps to implement the LSTM Model are:
<li> Preparing the dataset for the LSTM Model on the basis of number of time steps.</li>
<li> Load the Dataset and Normalize the Data</li>
<li> Split the Data into Training and Testing Data</li>
<li> Reshape the Data</li>
<li> Build the LSTM Model</li>
<li> Train the Model</li>
<li> Test the Model on the Test Set</li>
<li> Calculate the RMSE Value</li>
<li> Make Predictions for the future values which is our main aim</li>

Here while predicting for future values, we take 3 scenarios into consideration:
##### 1: The given DataSet Has Already Been Stored And Predictions are Made
In this case we will directly predict the values using the already been stored in a CSV File and show this predictions to the user. 
 
##### 2: The Given DataSet Is 70 % Same as the already stored Dataset
In this case we will use the already stored model and train the model on this dataset only and make the predictions and then show the predictions to the user.

##### 3: The Given DataSet Is Completely New or No Dataset is Stored
In this case we will first find the best values of number of time steps and the Patience values and then train the dataset on the best model and then make the predictions and show the predictions to the user.


## Improving Model Accuracy 📈

To improve the accuracy of the training model, consider the following strategies:

  - Increase Number of Epochs: Training for more epochs can help the model learn better.
  - Change Optimizer: Experiment with different optimizers like Stochastic Gradient Descent, RMSprop, or AdaGrad.
  - Increase in Number of Layers: Adding more layers can help the model learn complex patterns.
  - Get More Data : More data can help the model generalize better.


## Getting Started 🚀

To get started with the Image Classification project:

  1. Clone the repository to your local machine.
  2. Install dependencies using pip install -r requirements.txt.
  3. Move inside the Glucose_Level_Prediction Folder and then cd into the Streamlit_Files Folder.
  4. Run the command streamlit run main.py
  5. Add the Dataset and see Magic Happening.


## Support and Feedback 📧

- For any issues, feedback, or queries related to the Glucose Level Prediction project, please drop a mail at soban1103@gmail.com


<b>Note :</b> For A New DataSet to be predicted the LSTM Model may take 20-30 minutes to train the model and make the predictions. So please be patient while the model is being trained. 