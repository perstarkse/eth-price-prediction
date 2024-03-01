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
from helper import get_price_history, get_formatted_dataset, get_predictions_not_eval, get_lag_llama_predictions

# Load environment variables from .env file
load_dotenv()

# Access environment variables
cg_api_key = os.getenv("COINGECKO_API_KEY")

response = get_price_history(coin_id="ethereum", vs_currency="usd", days="4000", api_key=cg_api_key)

print(len(response["prices"]))

dataset = get_formatted_dataset(response["prices"])

prediction_length = 7
num_samples = 10000

start_time = time.time()


# debug ouput
def eval_with_logs(dataset, prediction_length, num_samples):
    forecasts, tss = get_lag_llama_predictions(dataset=dataset, prediction_length=prediction_length, num_samples=num_samples)
    output_prediction = forecasts[-1].mean
    # latest_targets = tss[-1].values[:-prediction_length]
    latest_target = tss[-1].values[-prediction_length:].flatten()

    print("Output Prediction:", output_prediction)
    print("Latest Target:", latest_target)

    for i, (predicted, actual) in enumerate(zip(output_prediction, latest_target)):
        print("Index:", i, "| Predicted:", predicted, "| Actual:", actual)
        
    # output info
    elapsed_time = time.time() - start_time
    print("-------------------------------------------\n")
    print("Elapsed time: ", elapsed_time, "seconds")
    print(len(tss[0][0].values), "items in the dataset")
    print(tss[-1].values[-1], "latest value")
    print(len(forecasts[0].samples), "samples used")

def forecast_without_eval(dataset, prediction_length, num_samples):
    forecast = get_predictions_not_eval(dataset=dataset, prediction_length=prediction_length, num_samples=num_samples)
    elapsed_time = time.time() - start_time

    print("-------------------------------------------\n")
    print("Elapsed time: ", elapsed_time, "seconds")
    print(forecast["mean"])

# eval function
# eval_with_logs(dataset=dataset, prediction_length=prediction_length, num_samples=num_samples)

# forecast
forecast_without_eval(dataset=dataset, prediction_length=prediction_length, num_samples=num_samples)


exit()
