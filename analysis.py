import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf


def load_data(filepath, monthly=True):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.set_index('date', inplace=True)
    if monthly:
        df = df['streams'].resample('ME').count()
    return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def perform_analysis(ts):
    decomposition = seasonal_decompose(ts, period=7)

    plt.figure(figsize=(15, 10))
    plt.subplot(411)
    plt.title('Original Time Series')
    plt.plot(ts)
    plt.subplot(412)
    plt.title('Trend')
    plt.plot(decomposition.trend)
    plt.subplot(413)
    plt.title('Seasonal')
    plt.plot(decomposition.seasonal)
    plt.subplot(414)
    plt.title('Residual')
    plt.plot(decomposition.resid)
    plt.tight_layout()
    plt.show()

    adf_result = adfuller(ts.dropna())
    print('\nAugmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    acf_values = acf(ts.dropna(), nlags=40)
    plt.title('Autocorrelation Function')
    plt.stem(acf_values)

    plt.subplot(122)
    pacf_values = pacf(ts.dropna(), nlags=20)
    plt.title('Partial Autocorrelation Function')
    plt.stem(pacf_values)
    plt.tight_layout()
    plt.show()

    return decomposition


def main(filepath):
    ts = load_data(filepath)
    decomp = perform_analysis(ts)
    return decomp


if __name__ == "__main__":
    filepath = './data/streams.csv'
    decomposition = main(filepath)