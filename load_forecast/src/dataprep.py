# DATA PROCESSING METHODS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loads data
def load(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	load=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	load['hour']=pd.DatetimeIndex(load['time']).hour # new column for hours
	load['minute']=pd.DatetimeIndex(load['time']).minute # new column for minutes
	load=pd.pivot_table(load,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	load=load.applymap(lambda x:(x*1000)/60) # convert kW to Wh 
	load.sort_index(inplace=True) # sort entries (just in case)
	return load

# saves data to csv
def save(data,path):
	data.to_csv(path,header=True)
	
# combines minute time intervals into hours
def m2h(data,nan='keep',inplace=False):
	data_new=data if inplace else data.copy()
	if nan=='keep': # if we want to kken Nans
		data_new= data.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in minute time interval results in Nan for hour time interval
	return data_new

# flattens data, converts columns into a multiindex level
def flatten(data,inplace=False):
	data_new=data if inplace else data.copy()
	if not isinstance(data, pd.Series): data_new=data.stack()
	return data_new
	
# remove incomplete first and last days
def cut(data,inplace=False):
	data_new=data if inplace else data.copy()
	f,_=data_new.index.min() # first day
	l,_=data_new.index.max() # last day
	if len(data_new.loc[f])<24: # if first day is incomplete
		data_new.drop(f,level=0,inplace=True) # drop the whole day
	if len(data_new.loc[l])<24: # if last day is incomplete
		data_new.drop(l,level=0,inplace=True) # drop the whole day
	return data_new

# shifts data for time series forcasting
def shift(data,n_shifts=1,shift=1):
	data_shifted={} # lagged dataframes for merging
	for i in range(0,n_shifts+1): # for each time step
		label='targets' # label for target values
		if i!=n_shifts:label='t-{}'.format(n_shifts-i) # labels for patterns
		data_shifted[label]=data.shift(-i*shift) # add lagged dataframe
	res=pd.concat(data_shifted.values(),axis=1,join='inner',keys=data_shifted.keys()) # merge lagged dataframes
	return res.dropna() # TODO: handling missing values

# order timesteps from the oldest
def order(data, inplace=False):
	data_new=data if inplace else data.copy()
	data_new=data_new[sorted(data_new.columns,reverse=True,key=(lambda x:x[0]))] # sort first level of column multiindex in descending order
	return data_new
	
# split data into patterns & targets
def split_X_Y(data,target_label='targets'):
	X=data.select(lambda x:x[0] not in [target_label], axis=1) # everything not labelled "target" is a pattern, [0] refers to the level of multi-index
	Y=data[target_label] # targets
	return X, Y

# split data into train & test sets
def split_train_test(data, base=7,test_size=0.25): # in time series analysis order of samples usually matters, so no shuffling of samples
	idx=flr((1-test_size)*len(data),base) # calculate number of samples in train set 
	train,test =data[:idx],data[idx:] # split data into train & test sets
	return train,test

# split data into n datasets (according to weekdays)
def split(data,nsplits=7): 
	return {i:data.iloc[i::nsplits,:] for i in range(nsplits)} # return as a dictionary {offset:data}
	
# rounds down to the nearest multiple of base
def flr(x,base=7):
	return base*int(x/base)