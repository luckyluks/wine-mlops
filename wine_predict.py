import requests
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def fetch_data():
    print("downloading")
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    print("download done")
    # print(data.sample(n=1).to_json(orient='split'))
    return test_x

def predict(data, host):
    endpoint=host
    print(f"endpoint: {endpoint}")
    # print(TEST_DATA)
    headers = {'Content-Type': 'application/json'}
    # print(f"data: {json.dumps(data.sample(n=1).to_numpy())}")
    data_sample = data.sample(n=1).to_json(orient='split')
    print(f"data sample: {data_sample}")
    response = requests.post(endpoint, data=data_sample, headers=headers)
    print(f"Returned status code: '{response.status_code}'")
    print(f"Returned json: '{response}'")
    print(f"Returned json: '{response.text}'")


data = fetch_data()
predict(data=data, host="http://localhost:7476/invocations")