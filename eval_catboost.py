from catboost import CatBoostRegressor
import pandas as pd
from utils import eval_dataset_and_predictions, plot

def catboost_eval(prices, volumes, prediction_length):
    # API serves with the current price as well. 
    # Remove it if needed
    # del prices[-1]
    del volumes[-1]

    print(len(prices))
    print(len(volumes))

    # Set up timestamps
    timestamps = pd.to_datetime([pd.Timestamp(x[0], unit='ms') for x in prices])

    # Extract prices
    values = [x[1] for x in prices]
    volume_array = [x[1] for x in volumes]

    # Set up dataframe
    dataframe = pd.DataFrame({'date': timestamps, 'target': values, 'volume': volume_array})

    # Instantiate model
    model = CatBoostRegressor(task_type='CPU', loss_function='RMSE')    

    # Set hyperparameters
    param_grid = {
    'iterations': [250],
    'depth': [6],
    'learning_rate': [0.1],
    'l2_leaf_reg': [1]
    }

    # Perform grid search
    grid_res = model.grid_search(
        param_grid,
        dataframe['date'],
        dataframe['target'], 
        cv=3,
        refit=True,
        calc_cv_statistics=True
    )

    # Make predictions
    predicted = model.predict(dataframe)
    print(predicted[-1])

    print(grid_res['params'])

    # Evaluate the model
    eval_dataset_and_predictions(dataframe, predicted)
    # Plot it, check plots/plot.png
    plot(dataframe, predicted)

    # Create a date range
    future_date_range = pd.date_range(dataframe['date'].max(), periods=prediction_length, freq='D')

    # Create a new dataframe
    future_data = pd.DataFrame({
        'date': future_date_range,
        'volume': [dataframe['volume'].iloc[-1]] * prediction_length
    })

    future_predictions = model.predict(future_data, prediction_type='RawFormulaVal')
    print(future_predictions)

    return future_predictions

