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
def ets(train,test):
	train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
	r_train=ro.r.read.zoo('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=TRUE,sep=',',dec='.')
	
	pandas2ri.activate()
	forecast=importr('forecast') # forecast package
	xts=importr('xts',robject_translations = {".subset.xts": "_subset_xts2", "to.period": "to_period2"}) # extended time series package, specify conversion for conflicting names
	fitted=ro.r('fitted') # function exporting forecasts used while fitting model
	r_train=ro.DataFrame(train.reset_index()) # convert to R dataframe
	r_test=ro.DataFrame(test) # convert to R dataframe
	r_train_xts=xts(r_train,ro.POSIXct(r_train.rx2('date')))
	r_test_xts=xts(r_test,to_date(r_test.rx2('date')))
	fit_train=forecast.ets(r_train_xts) # find best model on train test
	fit_test=forecast.ets(r_test_xts,model=fit_train) # get predictions on test set
	return pandas2ri.ri2py(fitted(fit_train)),pandas2ri.ri2py(fitted(fit_test))



np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
dp.cut(load_raw,inplace=True) # remove leading & trailing Nans
targets=load_raw.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
targets.fillna(method='bfill',inplace=True)
targets.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_proc.csv',header=True,sep=',',decimal='.')


train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)
train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
test.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_test.csv',header=True,sep=',',decimal='.')




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




