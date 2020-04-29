import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

start = "14-04-2021 00:00:01"
end = "13-04-2015 00:00:00"

baseUrl = "***REMOVED***"
apikey = "***REMOVED***"
dst = "true"
status = "true"

def apicall(viewid, payload):
    queryDictionary = {"apikey":apikey, "start":start, "end":end, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
    reqUrl = baseUrl

    print("\nPulling data from API...")
    ("Url: %s \nAPI key: %s" % (baseUrl, apikey))

    first_key = True
    for key in queryDictionary:
        if first_key:
            reqUrl += "?%s=%s" % (key, queryDictionary[key])
            first_key = False
        else:
            reqUrl += "&%s=%s" % (key, queryDictionary[key])

    print("\nRequesting:", reqUrl)
    req = requests.post(reqUrl, json=payload)

    return req.json()['channel']['feeds'][0]['points'], req.elapsed.total_seconds()

print("\n==================================================================================\n")

# Times down call
viewid = "670"
payload = {"0":{"feedid":"oee_stopsec", "methode":"none"}}
jsonResponseD, elapsed = apicall(viewid, payload)
print("Times down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), elapsed))

## Production amount call
viewid = "694"
payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}
jsonResponseP, elapsed = apicall(viewid, payload)
print("Production amount API call got: %s points, call took: %s seconds" % (len(jsonResponseP), elapsed))



df = pd.DataFrame.from_dict(jsonResponseD)
#comments = df.pop('comment')
df = df.drop('pointid', axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['value'] = pd.to_numeric(df['value'])
df['comment'] = df['comment'].fillna(0)

dfDict = {'timestamp':[], 'category':[], 'comment':[]}
dfDict['timestamp'] = df['timestamp']

for comment in df['comment']:
    try:
        dfDict['category'].append(int(comment))
        dfDict['comment'].append("Uncategorised")
    except:
        c_json = json.loads(comment.replace("'", "\""))
        dfDict['category'].append(c_json['category'])

        # > 4 to exclude 'None' comments
        if len(str(c_json['comment'])) > 4:
            dfDict['comment'].append(str(c_json['comment']).strip('[]\''))
        else:
            dfDict['comment'].append("Uncategorised")

df = df.drop('comment', axis=1)
df = df.merge(pd.DataFrame.from_dict(dfDict), how='outer', on='timestamp')


#df = df.resample('D').sum()

#print(categories)
#print(comments)

dfP = pd.DataFrame.from_dict(jsonResponseP)
dfP = dfP.drop('pointid', axis=1)
dfP['timestamp'] = pd.to_datetime(dfP['timestamp'])
dfP['value'] = pd.to_numeric(dfP['value'])
dfP = dfP[dfP['value'] > 0]
dfP = dfP.rename(columns={'value': 'produced'})
dfP = dfP.set_index('timestamp')
dfP = dfP.resample('D').sum()

#df['comment'] =  (lambda x: x == "Planned repair" and 0 or 1)(df['comment'])
df = df.set_index('timestamp')

#df['timestamp'] = pd.datetime.isoformat(df['timestamp'])
#df.infer_objects().dtypes


# Unique strings
print(df.groupby('comment').sum())
print(df.groupby('category').sum())

print(df.dtypes)

dataset = tf.data.Dataset.from_tensor_slices(df)

#print(list(dataset.as_numpy_iterator()))



############################################################################################################################################

def pulldata2():

    start = "14-04-2021 00:00:01"
    end = "13-04-2015 00:00:00"

    baseUrl = "***REMOVED***"
    apikey = "***REMOVED***"
    dst = "true"
    status = "true"

    print("\n==================================================================================\n")
    print("\nPulling data from API...")
    print("Url: %s \nAPI key: %s" % (baseUrl, apikey))

    viewid = "670"

    queryDictionaryD = {"apikey":apikey, "start":start, "end":end, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
    payloadD = {"0":{"feedid":"oee_stopsec", "methode":"none"}}

    reqUrl = baseUrl
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

    queryDictionaryP = {"apikey":apikey, "start":start, "end":end, "dst":dst, "viewid":viewid, "status":status}
    payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}

    reqUrl = baseUrl
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

    return dfPred.values

def pulldata():
    
    ### API Data ###
    # Standard Values
    start = "14-04-2021 00:00:01"
    end = "13-04-2015 00:00:00"

    baseUrl = "***REMOVED***"
    apikey = "***REMOVED***"
    dst = "true"
    status = "true"


    print("\nPulling data from API...")
    print("Url: %s \nAPI key: %s" % (baseUrl, apikey))

    ## Times down call

    # Needed to get comment with data point
    # Views are different types of data pulls
    viewid = "670"

    queryDictionaryD = {"apikey":apikey, "start":start, "end":end, "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}
    payloadD = {"0":{"feedid":"oee_stopsec", "methode":"none"}}

    reqUrl = baseUrl
    for key in queryDictionaryD:
        if len(reqUrl) == len(baseUrl):
            reqUrl += "?%s=%s" % (key, queryDictionaryD[key])
        else:
            reqUrl += "&%s=%s" % (key, queryDictionaryD[key])

    reqD = requests.post(reqUrl, json=payloadD)

    jsonResponseD = reqD.json()['channel']['feeds'][0]['points']

    print("\nTimes down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), reqD.elapsed.total_seconds()))



    ## Production amount call
    viewid = "694"

    queryDictionaryP = {"apikey":apikey, "start":start, "end":end, "dst":dst, "viewid":viewid, "status":status}
    payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}

    reqUrl = baseUrl
    for key in queryDictionaryP:
        if len(reqUrl) == len(baseUrl):
            reqUrl += "?%s=%s" % (key, queryDictionaryP[key])
        else:
            reqUrl += "&%s=%s" % (key, queryDictionaryP[key])

    reqP = requests.post(reqUrl, json=payload)

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

    return dfPred.values


# Links
# https://stackabuse.com/converting-strings-to-datetime-in-python/
# https://docs.python.org/2/library/datetime.html
# https://docs.python.org/2/library/stdtypes.html#str.isdigit
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.get_params
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# https://projects.datacamp.com/projects/558
# https://www.tutorialspoint.com/python/python_loop_control.htm
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-query
# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
# https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
# https://stackoverflow.com/questions/2349991/how-to-import-other-python-files