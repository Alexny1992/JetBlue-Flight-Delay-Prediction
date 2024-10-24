import numpy as np 
from pyspark.sql import SparkSession
import prophet as fbprophet
from prophet.utilities import regressor_coefficients
from src.spark_initialize import spark
import pyspark.pandas as ps
from sklearn.metrics import mean_absolute_error, mean_squared_error

test_days = 31 
def training_split(df_prepared):
    training_set = df_prepared.limit(df_prepared.count() - test_days)
    training_set_pd = training_set.toPandas()
    print(training_set_pd)
    return training_set_pd
    
def test_split(df_prepared):
    test_set = df_prepared.tail(test_days)
    test_set = spark.createDataFrame(test_set)
    test_set_pd = test_set.toPandas()
    return test_set_pd

def fb_prophet(training_set_pd):
    m = fbprophet.Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=20.0,
        changepoint_prior_scale=0.1
    )
    m.add_regressor('arr_del15')
    m.add_regressor('weather_ct')

    m.fit(training_set_pd)
    
    return m 

def coefficient_regressor(m):
    print(regressor_coefficients(m))
    
def future_dataframe(m, df):    
    future = m.make_future_dataframe(periods=test_days, freq='d')
    df = df.reset_index()
    future = ps.merge(future, df, on='ds', how= 'left')
    future = future[['ds', 'arr_del15', 'weather_ct']]
    future['arr_del15'].fillna(method='ffill', inplace=True) 
    future['weather_ct'].fillna(method='ffill', inplace=True)
    return future

def forcast(m, future):
    forecast = m.predict(future)
    return forecast

def prediction(forecast):
    test_days = 31
    prediction = forecast['yhat'].tail(test_days)
    return prediction

def error_calculation(test_set, prediction):
    test_set = test_set['y'].to_numpy()
    mae = mean_absolute_error(test_set, prediction)
    rmse = np.sqrt(mean_squared_error(test_set, prediction))
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")