from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
import prophet as fbprophet

def run_cross_validation(model, horizon='31 days', initial='1500 days', parallel='processes'):
    """
    Perform cross-validation on a Prophet model.
    """
    df_cv = cross_validation(
        model=model,
        horizon=horizon,
        initial=initial,
        parallel=parallel
    )
    return df_cv

def calculate_performance_metrics(df_cv):
    """
    Calculate performance metrics based on cross-validated results.
    """
    df_p = performance_metrics(df_cv)
    mae = mean_absolute_error(df_cv['y'], df_cv['yhat'])
    rmse = np.sqrt(mean_squared_error(df_cv['y'], df_cv['yhat']))
    
    return df_p, mae, rmse

def evaluate_model(params, training_set_pd):
    """
    Fit and evaluate a Prophet model with given parameters.
    """
    # Initialize and configure the Prophet model
    m = fbprophet.Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive',
        seasonality_prior_scale=params['seasonality_prior_scale'],
        changepoint_prior_scale=params['changepoint_prior_scale']
    )
    
    # Add regressors
    m.add_regressor('arr_del15')
    m.add_regressor('weather_ct')
    
    # Fit the model
    m.fit(training_set_pd)
    
    # Perform cross-validation
    df_cv = run_cross_validation(model=m)
    
    # Measure the error
    error = np.sqrt(mean_squared_error(df_cv['y'], df_cv['yhat']))
    
    return error

def hyperparameter_tuning(training_set_pd):
    param_Grid = {
        'seasonality_prior_scale': [5, 10, 20],
        'holiday_prior_scale': [5, 10, 20],
        'changepoint_prior_scale': [0.01, 0.05, 0.1]
    }
    
    grid = ParameterGrid(param_Grid)
    rmse = []
    
    # Loop through the parameter grid and evaluate each set of parameters
    for params in grid:
        error = evaluate_model(params, training_set_pd)
        rmse.append(error)
    
    # Identify the best parameters based on RMSE
    best_index = np.argmin(rmse)
    best_params = grid[best_index]
    return best_params, rmse[best_index]
