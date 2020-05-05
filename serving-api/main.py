import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import logging
import util.data_puller as api
import util.difference_holder as dh
from json import JSONEncoder

serving_logger = logging.getLogger('serving-api')
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
serving_logger.setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config['DEBUG'] = True

history_size = 60

@app.route('/')
def hello_api():
    return "You've hit the API! Autch..."

@app.route('/v1/predict', methods=['GET'])
def api_predict():

    serving_logger.debug("Got API call for Predictor!")

    if 'apikey' in request.args and 'channel_id' in request.args and 'to' in request.args:
        try:
            apikey = request.args['apikey']
            channel_id = int(request.args['channel_id'])
            datetime_to = datetime.datetime.strptime(request.args['to'], "%Y-%m-%d %H:%M:%S")

            # NOTE: Take the day before to ensure only completed days
            datetime_to = datetime_to - datetime.timedelta(1)
            datetime_from = datetime_to - datetime.timedelta(history_size)
        except:
            return "Not allowed."

    else:
        return "Missing parameters."

    req_url = os.environ['API_BASE_URL'] + '/' + str(channel_id) + '/' + os.environ['API_F']

    train = api.apicallv3(history_size, req_url, apikey, datetime_to.strftime("%Y-%m-%d %H:%M:%S"), datetime_from.strftime("%Y-%m-%d %H:%M:%S"))

    X = list(train.as_numpy_iterator())

    pred_amount = len(X)
    threshold = 3

    differ = dh.DifferenceHolder(threshold, serving_logger)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    for i in range(pred_amount):

        tcx = [[]]
        tcx[0] = X[i][0]

        data = json.dumps({"signature_name":"serving_default", "instances":tcx}, cls=NumpyArrayEncoder)
        headers = {"content-type":"application/json"}
        json_response = requests.post("http://predictor-service:8501/v1/models/predictor:predict", data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']

        true_value = X[i][1]

        single_prediction = predictions[0][0]

        differ.difference_calc(single_prediction, true_value, tcx[0])

        i += 1

    return json.dumps(predictions)

app.run(host='0.0.0.0', port=5000)