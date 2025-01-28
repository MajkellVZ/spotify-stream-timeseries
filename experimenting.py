import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt

from logger import MlflowLogger


def load_and_prepare_data(filepath, monthly=True):
    """
    Load Spotify streams time series data and prepare it for modeling

    Expected CSV columns:
    - date: timestamp of the record
    - streams: number of streams
    - (optional) additional features like day of week, month, artist, etc.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    if monthly:
        df = df['streams'].resample('ME').count()
    return pd.DataFrame(df)


def create_lagged_features(df, lag_features=12):
    """
    Create lagged features for time series prediction

    Parameters:
    - df: Input dataframe
    - lag_features: Number of lag steps to create

    Returns:
    - Dataframe with lagged features
    """
    for i in range(1, lag_features + 1):
        df[f'streams_lag_{i}'] = df['streams'].shift(i)

    df = df.dropna()

    return df

def prepare_ml_data(df, target_col='streams'):
    """
    Prepare data for machine learning

    Returns:
    - X: Features
    - y: Target variable
    """
    feature_cols = [col for col in df.columns if col not in ['date', target_col]]

    X = df[feature_cols]
    y = df[target_col]

    return X, y

def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV

    Returns:
    - Best estimator
    - Best parameters
    """
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    xgb_model = xgb.XGBRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance

    Returns:
    - Dictionary of performance metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    }

    return metrics

def plot_predictions(y_train, y_test, y_pred, title='Model Predictions'):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.plot(np.ones(len(y_test)) * y_train.mean(), label='Baseline', color='yellow')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Streams')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(filepath, lags=12):
    logger = MlflowLogger()

    with logger:
        logger.log_param('data_path', filepath)

        df = load_and_prepare_data(filepath)

        df = create_lagged_features(df, lag_features=lags)
        logger.log_params({'lags': lags})

        X, y = prepare_ml_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_model, best_params = tune_xgboost(X_train_scaled, y_train)
        logger.log_params(best_params)

        metrics = evaluate_model(best_model, X_test_scaled, y_test)
        logger.log_metrics(metrics)

        y_pred = best_model.predict(X_test_scaled)

        plot_predictions(y_train, y_test, y_pred, 'Spotify Streams Prediction')

        logger.log_model(best_model, 'model')
        logger.register_model('model')


if __name__ == "__main__":
    filepath = './data/streams.csv'
    main(filepath)
