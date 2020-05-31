import numpy as np
import pandas as pd
import datetime
import requests
import json
import time
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger('data-puller')

def apicall(viewid, req_url, payload, apikey, start, end):
    dst = "true"
    status = "true"

    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    queryDictionary = {"apikey":apikey, "start":start.strftime("%d-%m-%Y %H:%M:%S"), "end":end.strftime("%d-%m-%Y %H:%M:%S"), "dst":dst, "viewid":viewid, "status":status, "wherevalue":">0"}

    logger.info("Requesting data from API...")

    first_key = True
    for key in queryDictionary:
        if first_key:
            req_url += "?%s=%s" % (key, queryDictionary[key])
            first_key = False
        else:
            req_url += "&%s=%s" % (key, queryDictionary[key])

    logger.debug(f"Requesting: {req_url}")
    req = requests.post(req_url, json=payload)

    if req.status_code != 200:
        logger.warning("Failed to get OK response")

    if len(req.json()['channel']['feeds'][0]['points']) < 1:
        logger.error("No data from API call!")

    return req.json()['channel']['feeds'][0]['points'], req.elapsed.total_seconds()



def down_time_transformer(json_data):

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

    logger.info("Normalizing data...")

    df['times_down'] = df['times_down'].mask(df['times_down'] > 40)
    df['times_down'] = df['times_down'].fillna(df['times_down'].mean())
    
    if df.isna().sum().sum() > 0:
        logger.warning(f"Data contains {df.isna().sum().sum()} NaN values after comment handling")

    return df



def product_produced_transformer(json_data):

    dfP = pd.DataFrame.from_dict(json_data)
    dfP = dfP.drop('pointid', axis=1)
    dfP['timestamp'] = pd.to_datetime(dfP['timestamp'])
    dfP['value'] = pd.to_numeric(dfP['value'])

    dfP['value'] = dfP['value'].mask(dfP['value'] < 0)
    dfP['value'] = dfP['value'].mask(dfP['value'] > 5000)
    dfP['value'] = dfP['value'].fillna(dfP['value'].mean())
    
    if dfP.isna().sum().sum() > 0:
        logger.warning(f"Data contains {dfP.isna().sum().sum()} NaN values after data normalization")

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

    # Times down call
    viewid = "670"
    payload = {"0":{"feedid":"oee_stopsec", "methode":"none"}}
    jsonResponseD, elapsed = apicall(viewid, req_url, payload, apikey, start, end)
    logger.debug("Times down API call got: %s points, call took: %s seconds" % (len(jsonResponseD), elapsed))

    # Production amount call
    viewid = "694"
    payload = {"0":{"feedid":"p1_cnt","methode":"diff"}}
    jsonResponseP, elapsed = apicall(viewid, req_url, payload, apikey,  start, end)
    logger.debug("Production amount API call got: %s points, call took: %s seconds" % (len(jsonResponseP), elapsed))
    
    logger.info("Starting datahandling...")
    start_time =  time.perf_counter()

    df = down_time_transformer(jsonResponseD)
    
    dfComments = df.pop('comment')

    dfP = product_produced_transformer(jsonResponseP)

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
        logger.warning(f"Data contains {df.isna().sum().sum()} NaN values after midnight fill")
    
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

    if predictor_call:
        df = pd.DataFrame(newdata).drop(columns=[0])
    else:
        df = pd.DataFrame(newdata, columns=df.columns)
        df = df.rename(columns={'timestamp':'days_to_maintenance', 'comment':'maintenance'})

    if df.isna().sum().sum() > 0:
        logger.warning(f"Final dataset contains {df.isna().sum().sum()} NaN values")

    logger.info("Finished datahandling!")
    logger.debug(f"Got a total of {len(df.values)} rows")
    logger.debug("Datahandling took: %s" % ((time.perf_counter() - start_time)))

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