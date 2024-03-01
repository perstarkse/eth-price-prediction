from MFLES.Forecaster import MFLES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import eval_dataset_and_predictions


def fit_and_predict(prices, timeframe):
    del prices[-1]

    timestamps = pd.to_datetime([pd.Timestamp(x[0], unit='ms') for x in prices])

    # Extract prices
    values = [x[1] for x in prices]

    # Create a DataFrame
    dataframe = pd.DataFrame({'date': timestamps, 'target': values})

    mfles = MFLES()
    # opt_params = mfles.optimize(dataframe['target'].values,
    #                       seasonal_period=1460,
    #                       test_size=len(prices) * 0.05,
    #                       n_steps=1, #number of train/test splits to make
    #                       step_size=10, #the number of periods to move each step
    #                       metric='smape', #should support smape, mse, mae, mape
    #                       )
    # print(opt_params)
    fitted = mfles.fit(dataframe["target"].values, seasonal_period=1460, smoother=True, trend_penalty=False)
    # fitted = mfles.fit(dataframe["target"].values, **opt_params)    
    predicted = mfles.predict(timeframe)

    plt.plot(np.append(fitted[-90:], predicted))
    plt.plot(dataframe['target'].values[-90:])
    plt.show()

    plt.savefig(f'plots/plot.png')

    eval_dataset_and_predictions(dataframe, fitted)

    return predicted


# def fit_and_predict(prices, timeframe):
#     del prices[-1]

#     timestamps = pd.to_datetime([pd.Timestamp(x[0], unit='ms') for x in prices])

#     # Extract prices
#     values = [x[1] for x in prices]

#     # Create a DataFrame
#     dataframe = pd.DataFrame({'date': timestamps, 'target': values})

#     mfles = MFLES()
#     mfles.fit(dataframe["target"].values, seasonal_period=1460)

#     # Get the predicted values and the conformal prediction intervals
#     predicted, upper_bounds, lower_bounds = mfles.conformal(dataframe['target'].to_numpy(), forecast_horizon=timeframe, n_windows=1, coverage=[0.9, 0.95, 0.99])

#     # Convert the upper and lower bounds to 1-dimensional arrays
#     upper_bounds = np.squeeze(upper_bounds)
#     lower_bounds = np.squeeze(lower_bounds)

#     # Plot the predicted values and the conformal prediction intervals
#     plt.plot(predicted, label="Predicted values")
#     plt.plot(upper_bounds, label="Upper bounds")
#     plt.plot(lower_bounds, label="Lower bounds")

#     # Fill the area between the upper and lower bounds
#     # plt.fill_between(range(len(predicted)), upper_bounds, lower_bounds, alpha=0.2, label="Conformal prediction intervals")

#     # Add a legend
#     plt.legend()

#     # Show the plot
#     plt.show()

#     # Save the plot to a file with a unique filename
#     plt.savefig(f'plots/plot.png')


#     return predicted

