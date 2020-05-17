import datetime
import logging
import requests
import json
import os
import numpy as np
import util.data_puller as api
from json import JSONEncoder
from fastapi import FastAPI
from pydantic import BaseModel

serving_logger = logging.getLogger('serving-api')
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
serving_logger.setLevel(logging.DEBUG)

app = FastAPI(title="Serving API", description="Serving version of the Predictor API.", version="1.2.1")

history_size = 60

class PredictionResult(BaseModel):
    date: str = None
    prediction_available: bool

class PredictionRequest(BaseModel):
    api_key: str
    channel_id: int
    prediction_date: datetime.date

class APIStatus(BaseModel):
    request_received: datetime.datetime = datetime.datetime.now()
    predictor_up_status: bool = False
    messsage: str = "You've hit the API! Autch..."

@app.get("/", response_model=APIStatus)
async def root():
    return APIStatus()

@app.post("/predict", response_model=PredictionResult)
async def predict(prediction_request : PredictionRequest):

    serving_logger.debug(f"Got API call for Predictor!")

    apikey = prediction_request.api_key
    channel_id = prediction_request.channel_id
    datetime_to = datetime.datetime.strptime(f"{prediction_request.prediction_date} 23:59:59", "%Y-%m-%d %H:%M:%S")

    # NOTE: Take the day before to ensure only completed days
    datetime_to = datetime_to - datetime.timedelta(1)

    # NOTE: Add 1 second to get the start of the next day so it matches history_size.
    datetime_from = (datetime_to - datetime.timedelta(history_size)) + datetime.timedelta(seconds=1)

    req_url = os.environ['API_BASE_URL'] + '/' + str(channel_id) + '/' + os.environ['API_F']

    try:
        train = api.apicallv3(history_size, req_url, apikey, datetime_to.strftime("%Y-%m-%d %H:%M:%S"), datetime_from.strftime("%Y-%m-%d %H:%M:%S"))
    except:
        return PredictionResult(prediction_available=False)

    X = list(train.as_numpy_iterator())

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    tcx = [[]]
    tcx[0] = X[0][0]

    data = json.dumps({"signature_name":"serving_default", "instances":tcx}, cls=NumpyArrayEncoder)
    headers = {"content-type":"application/json"}
    json_response = requests.post("http://predictor-service:8501/v1/models/predictor:predict", data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    single_prediction = predictions[0][0]

    return PredictionResult(date=single_prediction,prediction_available=True)