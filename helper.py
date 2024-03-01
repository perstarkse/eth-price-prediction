from itertools import islice
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from lag_llama.gluon.estimator import LagLlamaEstimator
from dotenv import load_dotenv
import os
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
import requests
from gluonts.dataset.common import ListDataset
from pandas import Timestamp
import json
import time

def get_predictions_not_eval(dataset, prediction_length, num_samples=100):
    ckpt = torch.load("lag-llama.ckpt", map_location=torch.device('cuda:0')) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32, # Should not be changed; this is what the released Lag-Llama model was trained with

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        batch_size=1,
        num_parallel_samples=100
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    
    forecast = next(predictor.predict(dataset=dataset, num_samples=num_samples))

    return {
        "mean": forecast.mean,
        "median": forecast.median,
        "start_date": forecast.start_date
    }

def get_lag_llama_predictions(dataset, prediction_length, num_samples=100, ):
    ckpt = torch.load("lag-llama.ckpt", map_location=torch.device('cuda:0')) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32, # Should not be changed; this is what the released Lag-Llama model was trained with

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        batch_size=1,
        num_parallel_samples=100
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss

def get_price_history(coin_id: str, vs_currency: str, days: str, api_key: str):
    # Base URL of the CoinGecko API
    root_url = "https://api.coingecko.com/api/v3"

    # Endpoint for historical market data
    endpoint = "/coins/{}/market_chart".format(coin_id)

    # Construct the query parameters
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily",  # Data interval (daily)
        "precision": "full"   # Full precision for currency price value
    }
    headers = {
        "x-cg-demo-api-key": api_key
    }
    # Make the API call
    response = requests.get(root_url + endpoint, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        # Print error message if request failed
        print("Error:", response.text)
        return None
    
def get_formatted_dataset(prices):
    del prices[-1]

    timestamps = pd.to_datetime([pd.Timestamp(x[0], unit='ms') for x in prices])

    # Extract prices
    values = [x[1] for x in prices]

    # Create a DataFrame
    data = pd.DataFrame({'date': timestamps, 'target': values})

    # Set the index to the datetime column without resampling
    data.set_index('date', inplace=True)

    # Optionally, you can ensure the index is sorted
    data.sort_index(inplace=True)

    # Create ListDataset
    start = data.index[0]
    time_series = data['target'].values

    return ListDataset([{'target': time_series, 'start': start}],
                        freq='1D')
