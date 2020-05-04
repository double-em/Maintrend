import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json

import flask
from flask import request, jsonify

base_url = os.environ['API_BASE_URL'] + '/' + os.environ['API_CHANNEL'] + '/' + os.environ['API_F']
dst = "true"
status = "true"

print("Got API call for Datahandler!")

apikey = os.environ['API_KEY']
channel_id = 20
datetime_from = datetime.datetime.strptime("13-04-2019 00:00:00", "%d-%m-%Y %H:%M:%S")
datetime_to = datetime.datetime.strptime("14-04-2020 00:00:01", "%d-%m-%Y %H:%M:%S")
history_size = (datetime_to - datetime_from).days
print("History size:", history_size)

viewid = "670"

string_to = datetime_to.strftime("%d-%m-%Y %H:%M:%S")
string_from = datetime_from.strftime("%d-%m-%Y %H:%M:%S")

queryDictionaryD = {"apikey":apikey, "start":string_to, "end":string_from, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
payloadD = {"0":{"feedid":"oee_stopsec", "methode":"none"}}

reqUrl = base_url
first_key = True
for key in queryDictionaryD:
    if first_key:
        reqUrl += "?%s=%s" % (key, queryDictionaryD[key])
        first_key = False
    else:
        reqUrl += "&%s=%s" % (key, queryDictionaryD[key])

print("\nRequesting:", reqUrl)
reqD = requests.post(reqUrl, json=payloadD)

jsonResponseD = reqD.json()['channel']['feeds'][0]['points']

print("Times down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), reqD.elapsed.total_seconds()))



## Production amount call
viewid = "694"

queryDictionaryP = {"apikey":apikey, "start":string_to, "end":string_from, "dst":dst, "viewid":viewid, "status":status}
payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}

reqUrl = "%s/%s/feeds2" % (baseUrl, channel_id)
first_key = True
for key in queryDictionaryP:
    if first_key:
        reqUrl += "?%s=%s" % (key, queryDictionaryP[key])
        first_key = False
    else:
        reqUrl += "&%s=%s" % (key, queryDictionaryP[key])

reqP = requests.post(reqUrl, json=payload)

print("\nRequesting:", reqUrl)
jsonResponseP = reqP.json()['channel']['feeds'][0]['points']

print("Production amount API call got: %s points, call took: %s seconds" % (len(jsonResponseP), reqP.elapsed.total_seconds()))



# Datahandling for production amount call
print("\nPreprocessing data (Step 1)...")
dictsProduct = {"timestamp":[], "value":[]}

for point in jsonResponseP:
    value = int(point['value'])
    if value > 0:
        timestampPoint = datetime.datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S").date()
        dictsProduct['timestamp'].append(timestampPoint)
        dictsProduct['value'].append(value)

dfProduct = pd.DataFrame(data=dictsProduct).sort_values(by=['timestamp']).reset_index(drop=True)

# Datahandling for Times Down Call
dicts = {"category":[], "comment":[], "timestamp":[], "value":[]}

for point in jsonResponseD:
    timestampPoint = datetime.datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S").date()
    if not 'comment' in point:
        dicts['category'].append(0)
        dicts['comment'].append("Uncategorised")
        dicts['timestamp'].append(timestampPoint)
        dicts['value'].append(int(point['value']))
    else:
        point['comment'] = str.replace(point['comment'], "[", "")
        point['comment'] = str.replace(point['comment'], "]", "")
        if len(point["comment"]) > 3:
            newString = "%s'%s'%s" % (point["comment"][:12], point["comment"][12:13], point["comment"][13:])
            commentString = str.replace(newString, "\'", "\"")
            comment = json.loads(commentString)
            dicts['category'].append(int(comment['category']))
            dicts['timestamp'].append(timestampPoint)
            dicts['value'].append(int(point['value']))

            if pd.isnull(comment['comment']) or comment['comment'] == "":
                dicts['comment'].append("Uncategorised")
            else:
                dicts['comment'].append(comment['comment'])

        else:
            if point['comment'].isdigit():
                dicts['category'].append(int(point['comment']))
                dicts['comment'].append("Uncategorised")
                dicts['timestamp'].append(timestampPoint)
                dicts['value'].append(int(point['value']))
            else:
                dicts['category'].append(0)
                dicts['comment'].append(point['comment'])
                dicts['timestamp'].append(timestampPoint)
                dicts['value'].append(int(point['value']))

dfDown = pd.DataFrame(data=dicts, ).sort_values(by=['timestamp']).reset_index(drop=True)

print("Preprocessing data (Step 2)...")

maintenance_timestamps = []

for row in dfDown.values:
    if row[1] == "Planned repair":
        used = False
        for time_field in maintenance_timestamps:
            if row[2] == time_field:
                used = True

        if used == False:
            maintenance_timestamps.append(row[2])
        
maintenance_timestamps.sort()

if len(maintenance_timestamps) < 1:
    print("No maintenances found in provided timeframe.")

predDict = {
    "timestamp":[],
    "day_of_week":[],
    "maintenance_day":[],
    "produced_today":[],
    "times_down_today":[],
    "amount_down_today":[],
    "days_to_maintenance":[]
}

last_maintenance_timestamp = maintenance_timestamps[0]

maintenance_index = 1

first_point = dfProduct['timestamp'][0]

last_point = dfProduct['timestamp'][len(dfProduct.values) - 1]

range_size = (last_point - first_point).days

print("Got %s days to process and %s day(s) with planned repair(s)..." % (range_size, len(maintenance_timestamps)))

total_to_process = range_size

for i in range(0, range_size):

    point_timestamp = first_point + datetime.timedelta(days=i)

    if point_timestamp > last_maintenance_timestamp and point_timestamp <= maintenance_timestamps[maintenance_index]:
        predDict['timestamp'].append(point_timestamp)
        predDict['day_of_week'].append(point_timestamp.weekday())

        produced = dfProduct[dfProduct['timestamp'] == point_timestamp]['value'].sum()
        predDict['produced_today'].append(produced)
        
        times_down = len(dfDown[dfDown['timestamp'] == point_timestamp].values)
        predDict['times_down_today'].append(times_down)
        
        amount_down = dfDown['value'].where(dfDown['timestamp'] == point_timestamp).sum()
        predDict['amount_down_today'].append(amount_down)

        days_to_maintenance_tmp = (maintenance_timestamps[maintenance_index] - point_timestamp).days
        predDict['days_to_maintenance'].append(days_to_maintenance_tmp)
        if days_to_maintenance_tmp < 0:
            print(maintenance_timestamps[maintenance_index], point_timestamp, days_to_maintenance_tmp)

        planned_today = 0

        for timestamp_tmp_key in dfDown.values:
            if timestamp_tmp_key[2] == point_timestamp and timestamp_tmp_key[1] == "Planned repair":
                planned_today += 1

        if planned_today > 0:
            predDict['maintenance_day'].append(1)
            last_maintenance_timestamp = maintenance_timestamps[maintenance_index]
            maintenance_index += 1

            if maintenance_index == len(maintenance_timestamps):
                break

        else:
            predDict['maintenance_day'].append(0)

        i += 1

        # ttt = i / total_to_process
        # sys.stdout.write("\r")
        # sys.stdout.write("[%-20s] %d%%" % ("="*int(20*ttt), 100*ttt))
        # sys.stdout.flush()

dfPred = pd.DataFrame(data=predDict)

print(dfPred.describe())
print("Got", len(dfPred.values), "rows total...")