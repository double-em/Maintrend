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

    print("Got API call for Predictor!")

    

app.run(host='0.0.0.0', port=5000)