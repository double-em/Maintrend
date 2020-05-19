import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import time
import logging
from sklearn.preprocessing import MinMaxScaler

data_puller_logger = logging.getLogger('data-puller')
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
data_puller_logger.setLevel(logging.DEBUG)

def apicall(viewid, req_url, payload, apikey, start, end):
    dst = "true"
    status = "true"

    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    queryDictionary = {"apikey":apikey, "start":start.strftime("%d-%m-%Y %H:%M:%S"), "end":end.strftime("%d-%m-%Y %H:%M:%S"), "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}

    data_puller_logger.info("Requesting data from API...")

    first_key = True
    for key in queryDictionary:
        if first_key:
            req_url += "?%s=%s" % (key, queryDictionary[key])
            first_key = False
        else:
            req_url += "&%s=%s" % (key, queryDictionary[key])

    data_puller_logger.debug(f"Requesting: {req_url}")
    req = requests.post(req_url, json=payload)

    if req.status_code != 200:
        data_puller_logger.warning("Failed to get OK response")

    if len(req.json()['channel']['feeds'][0]['points']) < 1:
        data_puller_logger.error("No data from API call!")

    return req.json()['channel']['feeds'][0]['points'], req.elapsed.total_seconds()



def data_getter(req_url, apikey, start, end):

    # Times down call
    viewid = "670"
    payload = {"0":{"feedid":"oee_stopsec", "methode":"none"}}
    jsonResponseD, elapsed = apicall(viewid, req_url, payload, apikey, start, end)
    data_puller_logger.debug("Times down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), elapsed))

    # Production amount call
    viewid = "694"
    payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}
    jsonResponseP, elapsed = apicall(viewid, req_url, payload, apikey,  start, end)
    data_puller_logger.debug("Production amount API call got: %s points, call took: %s seconds" % (len(jsonResponseP), elapsed))

    return jsonResponseD, jsonResponseP



def down_time_transformer(json_data, raw_data=False):

    df = pd.DataFrame.from_dict(json_data)
    df = df.drop('pointid', axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['value'] = pd.to_numeric(df['value'])
    df = df.rename(columns={'value': 'downtime'})
    df['comment'] = df['comment'].fillna(0)

    dfDict = {'timestamp':[], 'times_down':[],
    'comment':[]}

    dfDict['timestamp'] = df['timestamp']

    for comment in df['comment']:
        dfDict['times_down'].append(1)
        try:
            comment = int(comment)
            dfDict['comment'].append(0)
        except:
            c_json = json.loads(comment.replace("'", "\""))
            
            if "Planned repair" in str(c_json['comment']) or "Unplanned repair" in str(c_json['comment']):
                dfDict['comment'].append(1)
            else:
                dfDict['comment'].append(0)

    df = df.drop('comment', axis=1)
    df = df.merge(pd.DataFrame.from_dict(dfDict), how='outer', on='timestamp')
    df = df.set_index('timestamp')
    df = df.resample('D').sum()

    data_puller_logger.info("Normalizing data...")

    df['times_down'] = df['times_down'].mask(df['times_down'] > 40)
    df['times_down'] = df['times_down'].fillna(df['times_down'].mean())
    
    if df.isna().sum().sum() > 0:
        data_puller_logger.warning(f"Data contains {df.isna().sum().sum()} NaN values after comment handling")

    return df



def product_produced_transformer(json_data, raw_data=False):

    dfP = pd.DataFrame.from_dict(json_data)
    dfP = dfP.drop('pointid', axis=1)
    dfP['timestamp'] = pd.to_datetime(dfP['timestamp'])
    dfP['value'] = pd.to_numeric(dfP['value'])

    # Filling big values with mean instead of 0
    dfP['value'] = dfP['value'].mask(dfP['value'] < 0)
    dfP['value'] = dfP['value'].mask(dfP['value'] > 5000)
    dfP['value'] = dfP['value'].fillna(dfP['value'].mean())
    
    if dfP.isna().sum().sum() > 0:
        data_puller_logger.warning(f"Data contains {dfP.isna().sum().sum()} NaN values after data normalization")

    dfP = dfP.rename(columns={'value': 'produced'})
    dfP = dfP.set_index('timestamp')
    dfP = dfP.resample('D').sum()

    return dfP



# Calculate days to next maintenance
def last_main(dataset):
    dataset = dataset[::-1]
    first_main = True
    last_m = 0
    remove = []
    for i in range(len(dataset)):

        if first_main:
            if dataset[i][4] == 1:
                first_main = False
                last_m = dataset[i][0]
                dataset[i][0] = 0
            else:
                remove.append(i)

        else:
            if dataset[i][4] == 1:
                last_m = dataset[i][0]
                dataset[i][0] = 0
            else:
                dataset[i][0] = int((((last_m - dataset[i][0])/60)/60)/24)
        
        i += 1

    dataset = np.delete(dataset, remove, axis=0)

    return dataset[::-1]



def apicallv3(req_url, apikey, start, end, predictor_call=False, raw_data=False):

    jsonResponseD, jsonResponseP = data_getter(req_url, apikey, start, end)
    
    data_puller_logger.info("Starting datahandling...")
    start_time =  time.perf_counter()

    df = down_time_transformer(jsonResponseD, raw_data)
    
    dfComments = df.pop('comment')

    dfP = product_produced_transformer(jsonResponseP, raw_data)

    dfP = dfP.reset_index()
    df = df.reset_index()

    df = df.merge(dfP, how='outer', on='timestamp')
    df = df.merge(dfComments, how='outer', on='timestamp')
    df['comment'] = df['comment'].mask(df['comment'] > 0).fillna(1)
    df['timestamp'] = df['timestamp'].astype(int) / 10**9

    scaler = MinMaxScaler(feature_range=(0,1))

    # At midnight when the system runs but no downtime
    df = df.fillna(0)

    if df.isna().sum().sum() > 0:
        data_puller_logger.warning(f"Data contains {df.isna().sum().sum()} NaN values after midnight fill")
    
    data_arr = df.values

    # Remove the first days where they havent logged much maintenance
    # NOTE: Helped alot!
    if not predictor_call:
        first_index = 0
        for i in range(len(data_arr)):
            if data_arr[i][4] == 1:
                break
            first_index += 1
        newdata = last_main(data_arr[first_index:])
    else:
        newdata = data_arr

    if not raw_data:
        newdata[:,1:-1] = scaler.fit_transform(newdata[:,1:-1])

    data_puller_logger.info("Correcting datatypes...")
    if predictor_call:
        df = pd.DataFrame(newdata).drop(columns=[0])
    else:
        df = pd.DataFrame(newdata, columns=df.columns)#.astype({0:'int32', 1:'float32', 2:'float32', 3:'float32', 4:'int32'})
        df = df.rename(columns={'timestamp':'days_to_maintenance', 'comment':'maintenance'})

    if df.isna().sum().sum() > 0:
        data_puller_logger.warning(f"Final dataset contains {df.isna().sum().sum()} NaN values")

    data_puller_logger.info("Finished datahandling!")
    data_puller_logger.debug(f"Got a total of {len(df.values)} rows")
    data_puller_logger.debug("Datahandling took: %s" % ((time.perf_counter() - start_time)))

    return df

# Links
# https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/0d390920-2ece-46ac-adea-25f4f54265f7
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c04_time_windows.ipynb#scrollTo=hzp7RD6_8OIY
# https://www.tensorflow.org/guide/data_performance#prefetching
# https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
# https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/blob/d3bb6b7dac10159b2e8a0a21fbf27e0078c3321b/StockPricesPredictionProject/pricePredictionLSTM.py#L20
# https://www.bioinf.jku.at/publications/older/2604.pdf
# https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#list_files
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html?highlight=resample#pandas.DataFrame.resample
# https://www.tensorflow.org/guide/data#time_series_windowing
# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#load_data_using_tfdatadataset
# https://www.tensorflow.org/tutorials/structured_data/time_series
# https://www.tensorflow.org/tfx/tutorials/transform/census#create_a_beam_transform_for_cleaning_our_input_data
# https://www.tensorflow.org/guide/keras/train_and_evaluate#api_overview_a_first_end-to-end_example

############################################################################################################################################

def pulldata2():

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

    start_time =  time.perf_counter()

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
    print("Datahandling took: %s" % ((time.perf_counter() - start_time)))
    print("\n==================================================================================\n")

    return dfPred.values

def pulldata():
    
    ### API Data ###
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