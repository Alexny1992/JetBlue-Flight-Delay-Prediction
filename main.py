from data_loading import data_loading, filter_data, prepare_data, convert_to_pandas
import numpy as np
import os
from pyspark.sql import SparkSession
from data_visualization import plot_data_matplotlib, plot_data_d3, plot_forecast, plot_forecast_components
from prophet_module import training_split, fb_prophet, test_split, coefficient_regressor, future_dataframe, forcast, prediction, error_calculation
from cross_validation import run_cross_validation, calculate_performance_metrics, evaluate_model, hyperparameter_tuning

file_path = '/Users/alexwang/Downloads/Airline Delay Cause.csv'
data = data_loading(file_path)

carrier_name = 'JetBlue Airways'
airport_code = 'JFK'
df_filtered = filter_data(data, carrier_name, airport_code)
df_spark_prepared = prepare_data(df_filtered) # preparing 'ds' and 'y' for fbprophet model 

# Displaying the DataFrame 
df_spark_prepared.show()

# EDA 
df_spark_prepared.describe().show() # understanding the dataset

# Plotting out a time series of df
df_pandas = convert_to_pandas(df_spark_prepared)
# plot_data_matplotlib(df_pandas, 'ds', 'y', 'JetBlue Flight Delay Arrival Ratio Over Time', 'Date', 'Delay_Arrival_Ratio')

# Training set and Test set split
training_set_pd = training_split(df_spark_prepared)
test_set_pd = test_split(df_spark_prepared)

# Facebook Prophet
m = fb_prophet(training_set_pd)
future = future_dataframe(m, df_pandas)
forcast = forcast(m, future)
print(forcast)
prediction = prediction(forcast)
error = error_calculation(test_set_pd, prediction)
plot_forecast(m, forcast)
# plot_forecast_components(m, forcast)


# Using cross validation to see how to refine parameters
df_cv = run_cross_validation(model = m)
df_p, mae, rmse = calculate_performance_metrics(df_cv)
best_params, best_rmse = hyperparameter_tuning(training_set_pd)
print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)
evaluate_model(best_params, training_set_pd)
