#!/usr/bin/env python3
import os
import pandas as pd
from sklearn import linear_model
import pickle

# training and saving some model
reg_model = linear_model.LinearRegression()
reg_model.fit([[2.,2.,5.], [1.,1.,5.], [3.,3.,2.]], [0.,0.,1.])
pickle.dump(reg_model, open('model/model.pkl', 'wb'))
print("Model training completed")

