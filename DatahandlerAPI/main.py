import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json

import flask
from flask import request, jsonify

# app = flask.Flask('DatahandlerAPI')
app = flask.Flask(__name__)
app.config['DEBUG'] = True

baseUrl = "***REMOVED***"
dst = "true"
status = "true"

@app.route('/')
def hello_api():
    return "You've hit the API! Autch..."

@app.route('/v1/datapull', methods=['GET'])
def api_datapull_all():

    print("Got API call!")

    if 'apikey' in request.args and 'channel_id' in request.args and 'from' in request.args and 'to' in request.args:
        try:
            apikey = request.args['apikey']
            channel_id = int(request.args['channel_id'])
            datetime_from = request.args['from']
            datetime_to = request.args['to']
        except:
            return "Not allowed." + datetime_from + " " + request.args['to']
        
    else:
        return "Missing parameters."

    viewid = "670"

    queryDictionaryD = {"apikey":apikey, "start":datetime_to, "end":datetime_from, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
    payloadD = {"0":{"feedid":"oee_stopsec", "methode":"none"}}

    reqUrl = "%s/%s/***REMOVED***" % (baseUrl, channel_id)
    first_key = True
    for key in queryDictionaryD:
        if first_key:
            reqUrl += "?%s=%s" % (key, queryDictionaryD[key])
            first_key = False
        else:
            reqUrl += "&%s=%s" % (key, queryDictionaryD[key])

    print("Requesting:", reqUrl)
    reqD = requests.post(reqUrl, json=payloadD)

    jsonResponseD = reqD.json()['channel']['feeds'][0]['points']

    print("\nTimes down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), reqD.elapsed.total_seconds()))



    ## Production amount call
    viewid = "694"

    queryDictionaryP = {"apikey":apikey, "start":datetime_to, "end":datetime_from, "dst":dst, "viewid":viewid, "status":status}
    payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}

    reqUrl = "%s/%s/***REMOVED***" % (baseUrl, channel_id)
    first_key = True
    for key in queryDictionaryP:
        if first_key:
            reqUrl += "?%s=%s" % (key, queryDictionaryP[key])
            first_key = False
        else:
            reqUrl += "&%s=%s" % (key, queryDictionaryP[key])

    reqP = requests.post(reqUrl, json=payload)

    print("Requesting:", reqUrl)
    jsonResponseP = reqP.json()['channel']['feeds'][0]['points']

    print("Production amount API call got: %s points, call took: %s seconds" % (len(jsonResponseP), reqP.elapsed.total_seconds()))



    # Datahandling for production amount call
    print("\nPreprocessing data (Step 1)...")
    dictsProduct = {"timestamp":[], "value":[]}

    for point in jsonResponseP:
        value = int(point['value'])
        if value > 0:
            timestampPoint = datetime.datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S")
            dictsProduct['timestamp'].append(timestampPoint)
            dictsProduct['value'].append(value)

    dfProduct = pd.DataFrame(data=dictsProduct)



    # Datahandling for Times Down Call
    dicts = {"category":[], "comment":[], "timestamp":[], "value":[]}

    for point in jsonResponseD:
        timestampPoint = datetime.datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S")
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

    df = pd.DataFrame(data=dicts)

    print("Preprocessing data (Step 2)...")

    maintenance_timestamps = []

    for row in df.values:
        if row[1] == "Planned repair":
            maintenance_timestamps.append(row[2])

    if len(maintenance_timestamps):
        return "No maintenances found in provided timeframe."

    predDict = {
        "day_of_week":[], 
        "days_since_last_maintenance":[], 
        "total_downs_slm":[], 
        "total_downtime_slm":[], 
        "product_produced_slm":[], 
        "days_to_maintenance":[]}

    last_maintenance_timestamp = maintenance_timestamps[0]

    maintenance_index = 0
    row_index = 0
    total_downs_slm_tmp = 0
    total_downtime_slm_tmp = 0

    skipped_repairs = 0

    total_to_process = len(df.values)
    processing_index = 0

    for row in df.values:
        timestamp_next_maintenance = maintenance_timestamps[maintenance_index]

        processing_index += 1

        if maintenance_timestamps[0].date() > row[2].date():
            continue

        if row[1] == "Planned repair":
            maintenance_index += 1

            if last_maintenance_timestamp.date() == row[2].date():
                # print("Skipped same day Planned repair...", row[2].date())
                skipped_repairs += 1
                continue

            # print("Planned repair at:", row[2].date())
            predDict["days_since_last_maintenance"].append(0)
            last_maintenance_timestamp = row[2]
            total_downs_slm_tmp = 0
            total_downtime_slm_tmp = 0
            
        else:
            predDict["days_since_last_maintenance"].append((row[2] - last_maintenance_timestamp).days)
            total_downs_slm_tmp += 1
            total_downtime_slm_tmp += row[3]

        
        predDict["day_of_week"].append(row[2].weekday())
        predDict["total_downs_slm"].append(total_downs_slm_tmp)
        predDict["total_downtime_slm"].append(total_downtime_slm_tmp)

        produced_items = dfProduct.query("timestamp <= @row[2] & timestamp >= @last_maintenance_timestamp")
        summed = produced_items['value'].sum()

        predDict["product_produced_slm"].append(summed)

        days_to_maintenance_tmp = (timestamp_next_maintenance - row[2]).days
        predDict["days_to_maintenance"].append(days_to_maintenance_tmp)
        
        row_index += 1
        ttt = processing_index / total_to_process
        sys.stdout.write("\r")
        sys.stdout.write("[%-20s] %d%%" % ("="*int(20*ttt), 100*ttt))
        sys.stdout.flush()

        if maintenance_index == len(maintenance_timestamps):
            sys.stdout.write("\r")
            sys.stdout.write("[%-20s] 100%%" % ("="*20))
            sys.stdout.flush()
            break

    dfPred = pd.DataFrame(data=predDict)


    print("\n\nTotal skipped same day repairs:", skipped_repairs)
    print("Got", len(dfPred.values), "rows total...")

    return dfPred.to_json()

app.run(host='0.0.0.0', port=5000)