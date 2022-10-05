# wine mlops


## running instructions

1. run the mlflow docker compose stack before
2. create a project in mlflow ui with the name "wine"
3. create new venv or use existing
4. install dependencies from requirements.txt
5. adapt variables in and run the training script: wine_training.py
6. adapt variables in and run the deployment script: wine_deploy.py
7. adapt variables in and run the prediction script (either single prediction or continously): wine_predict.py or wine_predict_continously.py
8. ...