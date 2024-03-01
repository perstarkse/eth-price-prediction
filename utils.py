from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

def symmetric_mean_absolute_percentage_error(actual, predicted):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))

def eval_dataset_and_predictions(dataframe, predicted) -> None:
    rmse = np.sqrt(mean_squared_error(dataframe['target'].values, predicted))
    mae = mean_absolute_error(dataframe['target'].values, predicted)
    mape = mean_absolute_percentage_error(dataframe['target'].values, predicted)
    smape = symmetric_mean_absolute_percentage_error(dataframe['target'].values, predicted)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("SMAPE:", smape)

def plot(dataframe, predicted) -> None:
    plt.plot(predicted[-90:])
    plt.plot(dataframe['target'].values[-90:])
    # plt.plot(predicted)
    # plt.plot(dataframe['target'])
    plt.show()

    plt.savefig(f'plots/plot.png')