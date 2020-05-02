import numpy as np
import requests

import json
from json import JSONEncoder

import importlib
api = importlib.import_module("API_Puller")
util = importlib.import_module("util")

from sklearn.preprocessing import MinMaxScaler



_back_in_time = 60
_step = 1
_target_size = 1

train = api.apicallv3(_back_in_time)

X = list(train.as_numpy_iterator())

pred_amount = len(X)
threshold = 3

differ = util.DifferenceHolder(threshold)

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

util.PrintFinal(differ)

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