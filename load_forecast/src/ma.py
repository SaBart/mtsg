'''MOVING AVERAGE'''

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import patsy
from sklearn.metrics import r2_score
from unittest.mock import inplace

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load=dp.load(path) # load data
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result

# moving average
for i in range(1,10):
	load_pred=load_with_nans.rolling(window=10).mean().shift(1)
	load_pred_tar=pd.concat({'pred':load_pred,'targets':load_with_nans},axis=1)
	load_pred_tar.dropna(inplace=True)
	r2_score(y_pred=load_pred_tar['pred'],y_true=load_pred_tar['targets'],multioutput='uniform_average')
	