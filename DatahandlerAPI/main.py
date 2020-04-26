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

@app.route('/v1/datapredictall', methods=['GET'])
def api_datapredict_all():

    print("\n==================================================================================\n")
    print("Got API call for Datahandler!\n")

    if 'apikey' in request.args and 'channel_id' in request.args and 'from' in request.args and 'to' in request.args:
        try:
            apikey = request.args['apikey']
            channel_id = int(request.args['channel_id'])
            datetime_from = datetime.datetime.strptime(request.args['from'], "%Y-%m-%d %H:%M:%S")
            datetime_to = datetime.datetime.strptime(request.args['to'], "%Y-%m-%d %H:%M:%S")

            history_size = (datetime_to - datetime_from).days
            if history_size < 1:
                return "Not allowed."
            print("History size: %s day(s)" % history_size)
        except:
            return "Not allowed."
        
    else:
        return "Missing parameters."

    viewid = "670"

    string_to = datetime_to.strftime("%d-%m-%Y %H:%M:%S")
    string_from = datetime_from.strftime("%d-%m-%Y %H:%M:%S")

    queryDictionaryD = {"apikey":apikey, "start":string_to, "end":string_from, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
    payloadD = {"0":{"feedid":"oee_stopsec", "methode":"none"}}

    reqUrl = "%s/%s/***REMOVED***" % (baseUrl, channel_id)
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

    reqUrl = "%s/%s/***REMOVED***" % (baseUrl, channel_id)
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

    print("Got %s day(s) to process and %s day(s) with planned repair(s)..." % (range_size, len(maintenance_timestamps)))
    print("Processing...")

    total_to_process = range_size - 1

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

            ttt = i / total_to_process
            sys.stdout.write("\r")
            sys.stdout.write("[%-20s] %d%%" % ("="*int(20*ttt), 100*ttt))
            sys.stdout.flush()

    dfPred = pd.DataFrame(data=predDict)
    print("\n\nGot", len(dfPred.values), "rows total... Returning to sender... Done!")
    print("\n==================================================================================\n")

    return dfPred.to_json()

@app.route('/v1/datapull', methods=['GET'])
def api_datapull_all():

    print("Got API call for Datahandler!")

    if 'apikey' in request.args and 'channel_id' in request.args and 'from' in request.args and 'to' in request.args:
        try:
            apikey = request.args['apikey']
            channel_id = int(request.args['channel_id'])
            datetime_from = datetime.datetime.strptime(request.args['from'], "%Y-%m-%d %H:%M:%S")
            datetime_to = datetime.datetime.strptime(request.args['to'], "%Y-%m-%d %H:%M:%S")
        except:
            return "Not allowed."
        
    else:
        return "Missing parameters."

    viewid = "670"

    string_to = datetime_to.strftime("%d-%m-%Y %H:%M:%S")
    string_from = datetime_from.strftime("%d-%m-%Y %H:%M:%S")

    queryDictionaryD = {"apikey":apikey, "start":string_to, "end":string_from, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
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

    if len(maintenance_timestamps) < 1:
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

# Ipaddress should be 0.0.0.0 or it binds to wrong endpoint and can't be reached.

# Links
# https://stackoverflow.com/questions/18458839/how-can-i-get-the-current-date-and-time-in-the-terminal-and-set-a-custom-command
# https://docs.docker.com/compose/networking/
# https://docs.docker.com/compose/gettingstarted/#step-5-edit-the-compose-file-to-add-a-bind-mount
# https://docs.docker.com/engine/reference/builder/#expose
# https://stackoverflow.com/questions/22111060/what-is-the-difference-between-expose-and-publish-in-docker
# https://stackoverflow.com/questions/41428382/connection-was-reset-error-on-flask-server
# http://containertutorials.com/docker-compose/flask-simple-app.html
# https://hub.docker.com/_/python?tab=description&page=1
# https://flask.palletsprojects.com/en/1.1.x/testing/#testing-json-apis
# https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask#creating-a-basic-flask-application
# https://www.codementor.io/@sagaragarwal94/building-a-basic-restful-api-in-python-58k02xsiq