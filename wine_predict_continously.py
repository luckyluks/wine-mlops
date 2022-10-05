import requests
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import time


PREDICTION_ENDPOINT="http://localhost:7476/invocations"
DATASET_ENDPOINT="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def fetch_data():
    time_before = time.time_ns()
    print("downloading ...")
    csv_url = DATASET_ENDPOINT
    data = pd.read_csv(csv_url, sep=";")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    print(f"download done in: {(time.time_ns()-time_before)/1000000000:.4f}s")
    data_sample = json.loads(data.sample(n=1).to_json(orient='split'))

    print(f"data sample (index={data_sample['index']}): \n    columns: {data_sample['columns']}\n    values: {data_sample['data']}")
    return test_x

def predict(data):
    headers = {'Content-Type': 'application/json'}
    data_sample = data.sample(n=1).to_json(orient='split')
    response = requests.post(PREDICTION_ENDPOINT, data=data_sample, headers=headers)
    return response

def setup():
    print(f"starting prediction loop at endpoint {PREDICTION_ENDPOINT}: {80*'='}")


data = fetch_data()
setup()

while True:
    time_before = time.time_ns()
    response = predict(data=data)
    print(f"Returned({response.status_code}): '{response.text}' in: {(time.time_ns()-time_before)/1000000:.4f}ms")
    time.sleep(1)