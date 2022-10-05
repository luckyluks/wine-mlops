from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from datetime import timedelta
from time import sleep
from random import randint

import numpy as np
import pandas as pd

import mlflow
import requests
import subprocess
import os
import logging

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logger")


###
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def fetch_data():
    logger.info("starting download")
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")
    logger.info("download done")
    return data
 

def train_model(data, mlflow_host, mlflow_experiment_name, alpha=0.5, l1_ratio=0.5):
    mlflow.set_tracking_uri(mlflow_host)
    logger.info(f"mlflow uri: '{mlflow_host}'")
    logger.info(f"mlflow version: {subprocess.run(['mlflow', '--version'], stdout=subprocess.PIPE).stdout}")
    
    # determine mlflow experiment id by name
    mlflow_experiment =  mlflow.get_experiment_by_name(name=mlflow_experiment_name)
    logger.info(f"mlflow experiment: '{mlflow_experiment.name}' (id={mlflow_experiment.experiment_id})")
 
    train, test = train_test_split(data)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    with mlflow.start_run(experiment_id=mlflow_experiment.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # simulate training
        sleep(randint(3,7))

        logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        logger.info("  RMSE: %s" % rmse)
        logger.info("  MAE: %s" % mae)
        logger.info("  R2: %s" % r2)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")


logger.info("starting ...")
data = fetch_data()
train_model(data=data, mlflow_host="http://localhost:5002", mlflow_experiment_name="wine", alpha=0.3, l1_ratio=0.3)
