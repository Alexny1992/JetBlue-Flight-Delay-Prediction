from pyspark.sql import SparkSession

# Create and configure Spark session
spark = SparkSession.builder \
    .appName("Airline_Delay_Cause") \
    .getOrCreate()