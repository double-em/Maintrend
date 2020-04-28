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

total_difference = 0
max_difference = 0
min_difference = 0

real_total_difference = 0
real_max_difference = 0
real_min_difference = 0

pred_amount = len(X)

threshold = 3
over_theshold_count = 0

for i in range(pred_amount):
    tcx = [[]]

    tcx[0] = np.array(X[i]).tolist()

    data = json.dumps({"signature_name":"serving_default", "instances":tcx})

    headers = {"content-type":"application/json"}
    json_response = requests.post("http://localhost:8501/v1/models/prod:predict", data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    
    
    true_value = y[i]

    single_prediction = predictions[0][0]
    pred_difference = abs(single_prediction - true_value)

    real_single_prediction = round(single_prediction)
    real_pred_difference = abs(real_single_prediction - true_value)



    total_difference += pred_difference

    if pred_difference > max_difference:
        max_difference = pred_difference

    if pred_difference < min_difference:
        min_difference = pred_difference



    real_total_difference += real_pred_difference
    
    if real_pred_difference > real_max_difference:
        real_max_difference = real_pred_difference

    if real_pred_difference < real_min_difference:
        real_min_difference = real_pred_difference



    i += 1
    if pred_difference > threshold:
        over_theshold_count += 1

        print("\n{:=^50}".format(" OVER TRESHOLD! "))
        print("Predictions: %s" % (len(predictions)))
        print("Predicted: %s(%s) \nActually: %s \nDifference: %s(%s)" % (
            single_prediction,
            real_single_prediction,
            true_value,
            pred_difference,
            real_pred_difference
        
        ))

        str_width = 10
        columns = [
            "maintenance_day",
            "produced_today",
            "times_down_today",
            "amount_down_today",
            "day_of_week"
        ]

        print("\nDataset:")
        for c in columns:
            print("{:{width}.8}".format(c, width=str_width), end="")
        print()
        
        for i in range(len(tcx[0])):
            j = tcx[0][i]
            for k in j:
                print("{:{width}.5}".format(str(k), width=str_width), end="")
            print()

total_mean_difference = total_difference / pred_amount
real_total_mean_difference = real_total_difference / pred_amount

print("\n========== Finished predictions! ==========")
print("{:22} {:.3f}".format("Total Loss:", total_difference))
print("{:22} {:.3f}".format("Total Mean Loss:", total_mean_difference))
print("{:22} {:.3f}".format("Maximum Loss:", max_difference))
print("{:22} {:.3f}".format("Minimum Loss:", min_difference))
print()
print("{:22} {}".format("Real Total Loss:", real_total_difference))
print("{:22} {:.3f}".format("Real Total Mean Loss:", real_total_mean_difference))
print("{:22} {}".format("Real Maximum Loss:", real_max_difference))
print("{:22} {}".format("Real Minimum Loss:", real_min_difference))
print()
print("{:22} {}".format("Threshold:", threshold))
print("{:22} {}".format("Total over threshold:", over_theshold_count))

### Links
# Cross Validation not used in NN: https://stackoverflow.com/questions/38164798/does-tensorflow-have-cross-validation-implemented-for-its-users
# https://www.dotnetperls.com/abs-python
# https://pyformat.info/
# https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
# https://stackoverflow.com/questions/27411631/inline-for-loop
# https://docs.python.org/3/library/string.html
# https://www.w3schools.com/python/python_lambda.asp