from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, concat, col, lit, month, date_format
import pandas as pd 
from src.spark_initialize import spark

def data_loading(file_path):
    try:
        data = spark.read.csv(file_path, header=True, inferSchema=True)
        print("File read successfully")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        print(data)

def filter_data(data, carrier_name, airport_code):
    df_filtered = data.filter((data['carrier_name'] == carrier_name) & (data['airport'] == airport_code))
    return df_filtered

def prepare_data(df_filtered):
    """
    Args:
        df_filtered (pyspark.sql.DataFrame): The filtered data.

    Returns:
        pyspark.sql.DataFrame: The prepared data.
    """
    
    df_filtered = df_filtered.withColumn('ds', to_date(concat(col('year'), lit('-'), col('month'))))
    df_filtered = df_filtered.withColumn('flight_ratio', col('arr_del15') / col('arr_flights') * 100)
    df_filtered = df_filtered.withColumnRenamed('flight_ratio', 'y')
    return df_filtered

def convert_to_pandas(df_filtered):
    """
    Converts a PySpark DataFrame to a Pandas DataFrame.

    Args:
        df_filtered (pyspark.sql.DataFrame): The PySpark DataFrame to convert.
        columns (list): The columns to select.

    Returns:
        pandas.DataFrame: The converted Pandas DataFrame.
    """
    
    pandas_df = df_filtered.toPandas()
    pandas_df = pandas_df.sort_values(by='ds')
    pandas_df['ds'] = pd.to_datetime(pandas_df['ds']) 
    return pandas_df