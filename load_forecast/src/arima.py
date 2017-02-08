'''HOLT-WINTER'S EXPONENTIAL SMOOTHING'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import patsy
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score

# calls libraries from R to find the best arima model 
def arima(train,test,freq=7):
	pandas2ri.activate()
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	fitted=ro.r('fitted') # function exporting forecasts used while fitting model
	r_train_ts=ts(train,frequency=freq) # construct R's ts object
	r_test_ts=ts(test,frequency=freq) # construct R's ts object
	fit_train=forecast.auto_arima(r_train_ts) # find best model on train test
	fit_test=forecast.Arima(r_test_ts,model=fit_train) # get predictions on test set
	train_pred=pd.Series(pandas2ri.ri2py(fitted(fit_train)),index=train.index) # reconstruct pandas DataFrame from R float vector
	test_pred=pd.Series(pandas2ri.ri2py(fitted(fit_test)),index=test.index) # reconstruct pandas DataFrame from R float vector
	return train_pred,test_pred

# searches for the best arima model for each hour separately
def arima_h(train,test,freq=7):
	train_pred=pd.DataFrame(data=None,index=train.index,columns=train.columns) # prepare dataframe for in sample predictions on train set
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe for out of sample prediction on test set
	for col in train: # for each hour
		train_pred[col],test_pred[col]=arima(train[col],test[col],freq=freq) # fill corresponding column with predictions
	return train_pred,test_pred

# searches for the best arima model for each hour & day of the week separately
def arima_wh(train,test,freq=52): # assume yearly seasonality, i.e. 52 weeks
	train_pred=pd.DataFrame(data=None,index=train.index,columns=train.columns) # prepare dataframe for in sample predictions on train set
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe for out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		train_day_pred,test_day_pred=arima_h(train_day,test_day,freq=freq) # predict for all hours of the respective day
		train_pred.iloc[i::7,:]=train_day_pred # fill corresponding rows with in sample predictions
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return train_pred,test_pred

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
dp.cut(load_raw,inplace=True) # remove leading & trailing Nans
targets=load_raw.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
targets.fillna(method='bfill',inplace=True)

train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)
train_pred,test_pred=arima_h(train,test)
r2_score(y_true=train,y_pred=train_pred,multioutput='uniform_average')
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')

train_pred,test_pred=arima_wh(train,test,freq=52)
r2_score(y_true=train,y_pred=train_pred,multioutput='uniform_average')
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')












train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
test.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_test.csv',header=True,sep=',',decimal='.')
targets.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_proc.csv',header=True,sep=',',decimal='.')



