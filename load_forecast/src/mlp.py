import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

# DATA PROCESSING METHODS

# loads data
def load_data(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	load=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	load['hour']=pd.DatetimeIndex(load['time']).hour # new culumn for hours
	load['minute']=pd.DatetimeIndex(load['time']).minute # new column for minutes
	load=pd.pivot_table(load,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	load=load.applymap(lambda x:(x*1000)/60) # convert kW to Wh 
	load.sort_index(inplace=True) # sort entries (just in case)
	return load

# remove incomplete first and last days
def cut_data(data_temp,inplace=False):
	if (inplace):data=data_temp
	else: data=data_temp.copy()
	f,_=data.index.min() # first day
	l,_=data.index.max() # last day
	if (len(data.loc[f])<24): # if first day is incomplete
		data.drop(f,level=0,inplace=True) # drop the whole day
	if (len(data.loc[l])<24): # if last day is incomplete
		data.drop(l,level=0,inplace=True) # drop the whole day
	return data

# shifts data for time series forcasting
def shift_data(data,nb_shifts=1,shift=7):
	data_lagged={} # lagged dataframes for merging
	for i in range(0,nb_shifts+1): # for each time step
		data_lagged[i-nb_shifts]=data.shift(-i*shift) # add lagged dataframe
	res=pd.concat(data_lagged.values(),axis=1,join='inner',keys=data_lagged.keys()) # merge lagged dataframes	
	return res.dropna()

# split data into X & Y
def split_X_Y(data):
	X=data.select(lambda x:x[0] not in [0], axis=1)
	Y=data[0]
	return X, Y

# split data into train & test sets
def split_train_test(data, test_size=0.2):
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import TimeSeriesSplit
	train, test =train_test_split(data, test_size=test_size)
	return train,test

# split data into 7 datasets according to weekdays
def split_week_days(data):
	Sun=data.iloc[::7, :] # simulation starts on Sunday 1 of January
	Mon=data.iloc[1::7, :]
	Tue=data.iloc[2::7, :]
	Wen=data.iloc[3::7, :]
	Thu=data.iloc[4::7, :]
	Fri=data.iloc[5::7, :]
	Sat=data.iloc[6::7, :]
	return Sun, Mon, Tue, Wen, Thu, Fri, Sat

# create & train basic NN model
def create_model(nb_in=24, nb_out=24, nb_hidden=10, nb_epoch=200, batch_size=10, activation='relu', loss='mean_squared_error', optimizer='adam'):
	from keras.models import Sequential
	from keras.layers.core import Dense
	model = Sequential() # FFN
	model.add(Dense(nb_hidden, input_dim=nb_in,activation=activation)) # input & hidden layers
	#model.add(Dropout({{uniform(0, 1)}})) # randomly set a number of inputs to 0 to prevent overfitting
	model.add(Dense(nb_out)) # output layer
	model.compile(loss=loss, optimizer=optimizer) # assemble network	
	return model

from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

seed=0 # fix seed for reprodicibility
np.random.seed(seed)
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load=load_data(path) # load data
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
# set grid search parameters and ranges
grid_space={'nb_hidden':[10,20,30],
			'nb_epoch':[500,1000,1500,2000],
			'batch_size':[1,5,10,20]
		}

for i in range(1,6): # optimize for number of time steps
	X,Y=split_X_Y(shift_data(load_with_nans,nb_shifts=i,shift=1).dropna()) # create patterns & targets in the correct format
	grid_space['nb_in']=[X.shape[1]] # workaround for enabling varying pattern lengths corresponding to the number of time steps
	model=KerasRegressor(build_fn=create_model,verbose=0) # create model template
	grid_setup = GridSearchCV(estimator=model, param_grid=grid_space, cv=TimeSeriesSplit(n_splits=3),n_jobs=1, scoring=make_scorer(r2_score,multioutput='uniform_average'), verbose=10) # set up the grid search
	grid_result = grid_setup.fit(X.as_matrix(), Y.as_matrix()) # fild best parameters
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # print best parameters	means = grid_result.cv_results_['mean_test_score']	stds = grid_result.cv_results_['std_test_score']	params = grid_result.cv_results_['params']	for mean, stdev, param in zip(means, stds, params):	print("%f (%f) with: %r" % (mean, stdev, param)) # print all sets of parameters

plt.plot(grid_result.best_estimator_.predict(X.as_matrix())[0])
