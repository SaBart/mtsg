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

# calls libraries from R to find the best exponential smoothing method 
def ets(train,test,freq=7):
	pandas2ri.activate()
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	fitted=ro.r('fitted') # function exporting forecasts used while fitting model
	r_train_ts=ts(train,frequency=freq) # construct R's ts object, assume weekly frequency
	r_test_ts=ts(test,frequency=freq) # construct R's ts object, assume weekly frequency
	fit_train=forecast.ets(r_train_ts) # find best model on train test
	fit_test=forecast.ets(r_test_ts,model=fit_train) # get predictions on test set
	train_pred=pd.Series(pandas2ri.ri2py(fitted(fit_train)),index=train.index) # reconstruct pandas DataFrame from R float vector
	test_pred=pd.Series(pandas2ri.ri2py(fitted(fit_test)),index=test.index) # reconstruct pandas DataFrame from R float vector
	return train_pred,test_pred

# searches for the best exponential smoothing model for each hour separately
def ets_h(train,test,freq=7):
	train_pred=pd.DataFrame(data=None,index=train.index,columns=train.columns)
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns)
	for col in train:
		train_pred[col],test_pred[col]=ets(train[col],test[col],freq=freq)
	return train_pred,test_pred

# searches for the best exponential smoothing model for each hour & day of the week separately
def ets_wh(train,test,freq=4):
	train_pred=pd.DataFrame(data=None,index=train.index,columns=train.columns) # in sample predictions on train set
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		train_day_pred,test_day_pred=ets_h(train_day,test_day,freq=freq) # predict for all hours of the respective day
		train_pred.iloc[i::7,:]=train_day_pred # fill corresponding rows with in sample predictions
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return train_pred,test_pred

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
dp.cut(load_raw,inplace=True) # remove leading & trailing Nans
targets=load_raw.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
targets.fillna(method='bfill',inplace=True)
targets.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_proc.csv',header=True,sep=',',decimal='.')


train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)
train_pred,test_pred=ets_h(train,test)
r2_score(y_true=train,y_pred=train_pred,multioutput='uniform_average')
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')

train_pred,test_pred=ets_wh(train,test,freq=4)
r2_score(y_true=train,y_pred=train_pred,multioutput='uniform_average')
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')












train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
test.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_test.csv',header=True,sep=',',decimal='.')




for col in train.columns:
	plt.figure()
	train[col].plot()
	train_pred[col].plot()
	plt.show()


series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
		  27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
		  26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
		  18,8,17,21,31,34,44,38,31,30,26,32]


def init_trend(series, slen):
	sum = 0.0
	for i in range(slen):
		sum += float(series[i+slen] - series[i])
	return sum / slen**2

init_trend(series, 12) # -0.7847222222222222
initial_trend(pd.DataFrame(series),12)




