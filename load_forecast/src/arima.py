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
from tqdm import tqdm
import time

# calls libraries from R to find the best arima model
def arima(train,test,hor=24,freq=24):
	pandas2ri.activate()
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	test_pred=test.copy()
	test_pred[:]=np.NaN
	for i in tqdm(range(len(test))): # for each sample in test set
		test_ts=ts(dp.flatten(pd.concat([train,test[:i]])),frequency=freq) # add a new day from test set to the current train set
		if i==0: # if  in the first iteration
			model=forecast.auto_arima(test_ts) # find best model on train test (i.e. original model)
			order=forecast.arimaorder(model) # retrieve the order of this best model
		else: # not the first iteration
			model=forecast.Arima(test_ts,order=order[0:3],seasonal=order[3:6]) # train Arima model on current (expanded) train set using order found on the original train set
		test_pred.iloc[i,:]=pd.Series(pandas2ri.ri2py(forecast.forecast(model,h=hor).rx2('mean'))) # predict all 24 hours for a new day & convert to pandas DataFrame
	return test_pred

# searches for the best arima model for horizontal predictions for each day of the week separately
def arima_hw(train,test,freq=52):
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		test_day_pred=arima(train_day,test_day,hor=24,freq=freq) # predict for all hours of the respective day
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return test_pred

# searches for the best arima model for vertical predictions for each hour separately
def arima_v(train,test,freq=7):
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set
	for col in train: # for each hour
		test_pred[col]=arima(train[col],test[col],hor=1,freq=freq) # fill corresponding column with predictions
	return test_pred

# searches for the best arima model for vertical predictions for each hour & day of the week 
def arima_vw(train,test,freq=52): # assume yearly seasonality, i.e. 52 weeks
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		test_day_pred=arima_v(train_day,test_day,freq=freq) # predict for all hours of the respective day
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return test_pred


np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
dp.cut(load_raw,inplace=True) # remove leading & trailing Nans
targets=dp.m2h(load_raw,nan='keep') # minutes to hours, preserving nans
targets.fillna(method='bfill',inplace=True) # fill nans withprevious values

train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)

# vertical
test_pred=arima_v(train,test)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/data/arima_v.csv')
# vertical week
test_pred=arima_vw(train,test,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/data/arima_vw.csv')
# horizontal
test_pred=arima(train,test)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/data/arima_h.csv')
# horizontal week
test_pred=arima_hw(train,test,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/data/arima_hw.csv')





train=dp.split(train,nsplits=7)[1]
test=dp.split(test,nsplits=7)[1]






train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
test.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_test.csv',header=True,sep=',',decimal='.')
targets.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_proc.csv',header=True,sep=',',decimal='.')

# calls libraries from R to find the best arima model 
def arima_old(train,test,freq=7):
	pandas2ri.activate() # activate interface between python and R 
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


def test(n=1000):
	for i in tqdm(range(n)):
		time.sleep(0.1)

