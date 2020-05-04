import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

pred_model = load_model("logs/pred_model")

# datetime_to = datetime.datetime.strptime("2020-01-22 21:05:23", "%Y-%m-%d %H:%M:%S")
# print(datetime_to)
# print(datetime_to.strftime("%d-%m-%Y %S:%M:%H"))

import flask
from flask import request, jsonify

base_url = os.environ['API_BASE_URL'] + '/' + os.environ['API_CHANNEL'] + '/' + os.environ['API_F']
apikey = os.environ['API_KEY']

# app = flask.Flask('DatahandlerAPI')
app = flask.Flask(__name__)
app.config['DEBUG'] = True

baseUrl = "https//datahandlerapi:5000/v1/datapredict"

history_size = 60 # Days
batch_size = 1

@app.route('/')
def hello_api():
    return "You've hit the API! Autch..."

@app.route('/v1/predict', methods=['GET'])
def api_datapull_all():

    print("Got API call for Predictor!")

    if 'apikey' in request.args and 'channel_id' in request.args and 'to' in request.args:
        try:
            apikey = request.args['apikey']
            channel_id = int(request.args['channel_id'])
            datetime_to = datetime.datetime.strptime(request.args['to'], "%Y-%m-%d %H:%M:%S")
            datetime_from = datetime_to - datetime.timedelta(history_size)
        except:
            return "Not allowed."

    else:
        return "Missing parameters."

    queryDictionary = {"apikey":apikey, "channel_id":channel_id, "to":datetime_to, "from":datetime_from}

    reqUrl = baseUrl
    first_key = True
    for key in queryDictionary:
        if first_key:
            reqUrl += "?%s=%s" % (key, queryDictionary[key])
            first_key = False
        else:
            reqUrl += "&%s=%s" % (key, queryDictionary[key])

    print("Requesting:", reqUrl)
    req = requests.post(reqUrl)

    df = pd.DataFrame.from_dict(req.json()).values

    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(df[:,:-1])

    data = []
    labels = []

    for i in range(0, len(dataset) - history_size, step):
        seq = dataset[i:i + history_size]
        label = target[i + history_size - 1]
        data.append(seq)
        labels.append(label)

    return data, labels

    pred_model.predict(X)

app.run(host='0.0.0.0', port=5000)