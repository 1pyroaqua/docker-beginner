#!/usr/bin/env python3
import os
from flask import Flask
from flask import request
import pandas as pd
from sklearn import linear_model
import pickle

app = Flask(__name__)

@app.route('/ping')
def index():
    return "true"

@app.route('/invocation', methods=['GET'])
def get_prediction():
    feature1 = float(request.args.get('f1'))
    feature2 = float(request.args.get('f2'))
    feature3 = float(request.args.get('f3'))
    loaded_model = pickle.load(open('model/model.pkl', 'rb'))
    prediction = loaded_model.predict([[feature1, feature2, feature3]])
    return str(prediction)

if __name__ == '__main__':
  app.run(port=5000,host='0.0.0.0')
  
