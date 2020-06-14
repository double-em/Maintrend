import pytest
import numpy as np
import requests
import os
import util.data_puller as api
import util.difference_holder as dh
import json
from json import JSONEncoder

from sklearn.preprocessing import MinMaxScaler

def test_api_predict(use_api_url, use_api_channel, use_api_f, use_api_key):

    base_url = os.environ['API_BASE_URL'] + '/' + os.environ['API_CHANNEL'] + '/' + os.environ['API_F']
    apikey = os.environ['API_KEY']

    _back_in_time = 60
    _step = 1
    _target_size = 1

    train = api.apicallv3(_back_in_time, base_url, apikey)

    X = list(train.as_numpy_iterator())

    pred_amount = len(X)
    threshold = 3

    differ = dh.DifferenceHolder(threshold)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    for i in range(pred_amount):

        new_new = json.dumps(X[i][0], cls=NumpyArrayEncoder)
        tcx = [[]]
        tcx[0] = X[i][0]

        data = json.dumps({"signature_name":"serving_default", "instances":tcx}, cls=NumpyArrayEncoder)
        headers = {"content-type":"application/json"}
        json_response = requests.post("http://localhost:8501/v1/models/prod:predict", data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']

        true_value = X[i][1]

        single_prediction = predictions[0][0]

        differ.difference_calc(single_prediction, true_value, tcx[0])

        i += 1

    dh.PrintFinal(differ)
    assert pred_amount > 0

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

# tcx = [[]]
# tcx[0] = df.values

# #print(tcx)

# data = json.dumps({"signature_name":"serving_default", "instances":tcx}, cls=NumpyArrayEncoder)
# #print(f"Data: {data}")
# headers = {"content-type":"application/json"}

# json_response = requests.post("http://172.22.0.2:8501/v1/models/predictor:predict", data=data, headers=headers)
# print(f"Json text: {json_response.text}")
# predictions = json.loads(json_response.text)['predictions']

# single_prediction = round(float(predictions[0][0]))

# date = datetime.datetime.now() + datetime.timedelta(single_prediction)
# date = date.date()

# print(date)

# print(single_prediction)

### Links
# https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#make_a_request_to_your_model_in_tensorflow_serving
# https://www.tensorflow.org/tfx/serving/docker
# https://www.tensorflow.org/guide/keras/train_and_evaluate


# Cross Validation not used in NN: https://stackoverflow.com/questions/38164798/does-tensorflow-have-cross-validation-implemented-for-its-users
# https://www.dotnetperls.com/abs-python
# https://pyformat.info/
# https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
# https://stackoverflow.com/questions/27411631/inline-for-loop
# https://docs.python.org/3/library/string.html
# https://www.w3schools.com/python/python_lambda.asp