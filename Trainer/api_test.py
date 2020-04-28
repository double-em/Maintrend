import numpy as np
import requests
import json
import importlib
api = importlib.import_module("API_Puller")
util = importlib.import_module("util")

from sklearn.preprocessing import MinMaxScaler



_back_in_time = 60
_step = 1
_target_size = 1

train = api.pulldata2()

def handle_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    
    data = []
    labels = []

    for i in range(0, len(dataset) - history_size, step):
        seq = dataset[i:i + history_size]
        label = target[i + history_size - 1]
        data.append(seq)
        labels.append(label)

    return data, labels

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(train[:,2:-1])

rescaledX = np.hstack((rescaledX, train[:,1:2]))

print("Making timestep sets (Step size: %s, History: %s days, Target value size: %s day(s))" % (_step, _back_in_time, _target_size))

X, y = handle_data(
    rescaledX, train[:, -1], 
    0, 
    len(train), 
    _back_in_time, 
    _target_size,
    _step, 
    single_step=True)



pred_amount = len(X)
threshold = 3

differ = util.DifferenceHolder(threshold)

for i in range(pred_amount):
    tcx = [[]]

    tcx[0] = np.array(X[i]).tolist()

    data = json.dumps({"signature_name":"serving_default", "instances":tcx})

    headers = {"content-type":"application/json"}
    json_response = requests.post("http://localhost:8501/v1/models/prod:predict", data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    true_value = y[i]

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