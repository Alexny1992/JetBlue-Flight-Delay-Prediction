# Flight Delay Prediction Using Spark and Facebook Prophet

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)  
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data](#data)
4. [Methodology](#methodology)

## Introduction

This project aims to predict future flight delays using the power of **Apache Spark** for big data processing and **Facebook Prophet** for time-series forecasting. By analyzing large datasets from the **U.S. Bureau of Transportation Statistics**, we uncover patterns and trends that contribute to delays, helping airlines optimize operations and improve customer experiences.

Key features include:
- Efficient data loading and preprocessing with Spark.
- In-depth exploratory data analysis (EDA).
- Predictive modeling using Prophet with custom regressors.
- Fine-tuning the model using cross-validation to minimize **Mean Squared Error (MSE)** and **Absolute Mean Error (MAE)**.

## Output
![Forcast](https://github.com/Alexny1992/JetBlue-Flight-Delay-Prediction/blob/main/image/Forcast.png)
![Forcast Components](https://github.com/Alexny1992/JetBlue-Flight-Delay-Prediction/blob/main/image/Forcast_Components.png)

## Getting Started

### Technology Used

- Apache Spark
- Python 3.x
- Matplotlib
- NumPy
- Seaborn
- Pandas (in Spark)
- Facebook Prophet
- Cross Validation (Scikit-learn)

## Data

- **Source:** [U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
  
This dataset contains flight delay data, including flight times, delays, and relevant metrics for analysis.

## Methodology

### 1. Data Extraction
- Extracted the dataset from the U.S. Bureau of Transportation.

### 2. Data Loading
- Created a function using Spark to load the data efficiently.

### 3. Exploratory Data Analysis (EDA)
- Conducted EDA using Spark to understand the dataset, identifying key trends and distributions.

### 4. Data Preparation
- Filtered and prepared the dataset for the **Facebook Prophet** model.

### 5. Train-Test Split
- Split the dataset into training and testing sets using Spark.

### 6. Model Implementation
- Applied **Facebook Prophet** with specified parameters to predict future flight delays.

### 7. Regressors Addition
- Enhanced the model by adding regressors that may influence flight delays.

### 8. Model Evaluation
- Used **Scikit-learn** to evaluate model performance with **Mean Squared Error (MSE)** and **Absolute Mean Error (MAE)**.

### 9. Visualization
- Visualized the forecast predictions to better understand the model outputs.

### 10. Hyperparameter Tuning
- Employed cross-validation and parameter tuning to optimize the model for the best **MSE** and **MAE
