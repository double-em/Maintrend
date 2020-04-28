import numpy as np
import requests
import json
import importlib
api = importlib.import_module("API_Puller")

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

tcx = [[]]

tcx[0] = np.array(X[0]).tolist()

data = json.dumps({"signature_name":"serving_default", "instances":tcx})

headers = {"content-type":"application/json"}
json_response = requests.post("http://localhost:8501/v1/models/prod:predict", data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

print("First prediction: %s Length: %s" % (predictions, len(predictions)))
print("Predicted: %s, Actually: %s" % (predictions[0][0], y[0]))