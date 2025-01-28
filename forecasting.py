import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


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

def get_best_model_by_mape(experiment_name: str, client: MlflowClient) -> Tuple[
    mlflow.pyfunc.PyFuncModel, int]:
    """
    Load the best model from an experiment based on the MAPE metric.

    Args:
        experiment_name: Name of the experiment to search for the best model
        client: MlflowClient object

    Returns:
        Tuple containing the best model and its MAPE score

    Raises:
        ValueError: If no runs are found with the MAPE metric
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    run_data = []
    for run in runs:
        metrics = run.data.metrics
        if 'MAPE' in metrics:
            run_data.append({
                'run_id': run.info.run_id,
                'MAPE': metrics['MAPE'],
                'lags': int(run.data.params.get('lags'))
            })

    if not run_data:
        raise ValueError("No runs found with MAPE metric")

    runs_df = pd.DataFrame(run_data)

    best_run = runs_df.sort_values('MAPE').iloc[0]
    best_mape = best_run['MAPE']
    best_run_id = best_run['run_id']
    lags = best_run['lags']

    try:
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Loaded best model from run {best_run_id} with MAPE: {best_mape:.4f}")

        run = client.get_run(best_run_id)
        metrics = run.data.metrics
        print("\nBest model metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return best_model, lags

    except Exception as e:
        raise Exception(f"Failed to load best model: {str(e)}")

def create_lagged_features(data, num_lags):
    """Create lagged features from time series data"""
    df = pd.DataFrame()
    for i in range(num_lags):
        df[f'lag_{i + 1}'] = data['streams'].shift(i)
    return df.dropna()

def get_last_day_of_month(date):
    """Get the last day of the month for a given date"""
    next_month = date + pd.DateOffset(months=1)
    return next_month - pd.DateOffset(days=next_month.day)

def forecast(X, horizon=7, model: Optional[mlflow.pyfunc.PyFuncModel] = None, num_lags: int = 12) -> pd.Series:
    """Forecast future values using a trained model"""

    if model is None:
        raise ValueError("Model not loaded. Call load_best_model() first")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=['streams']
    )

    last_window = X_scaled.iloc[-num_lags:]

    if len(last_window) < num_lags:
        raise ValueError(f"Not enough data points. Need at least {num_lags} months of data.")

    forecasts = []
    last_date = X.index[-1]
    forecast_dates = []
    current_window = last_window.copy()

    for i in range(horizon):
        X_input = create_lagged_features(current_window, num_lags)

        next_step = model.predict(X_input)[-1]
        forecasts.append(next_step)

        next_date = get_last_day_of_month(last_date + pd.DateOffset(months=i + 1))
        forecast_dates.append(next_date)

        new_row = pd.DataFrame({
            'streams': [next_step]
        }, index=[next_date])

        current_window = pd.concat([current_window.iloc[1:], new_row])

    forecast_series = pd.Series(
        forecasts,
        index=pd.DatetimeIndex(forecast_dates),
        name='forecast'
    )

    return forecast_series

def main():
    tracking_uri = "http://127.0.0.1:5000/"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    filepath = './data/streams.csv'
    data = load_and_prepare_data(filepath)

    model, num_lags = get_best_model_by_mape('forecasting', client)

    forecasts = forecast(
        data,
        horizon=12,
        model=model,
        num_lags=num_lags
    )

    forecasts.to_csv('./data/forecasts.csv')

if __name__ == "__main__":
    main()