'''
Created on 4 Dec 2016

@author: SABA
'''
import os
import pandas
import numpy
import sklearn
import matplotlib.pyplot as mp

from keras.models import Sequential
from keras.layers.core import Dense
#from numpy.distutils.conv_template import file
#from pandas.io.tests.parser import parse_dates


# PARAMETERS

time_ints=24 # number of time intervals
cols=numpy.arange(1,1+time_ints).tolist() # column names


# LOADING ENVIRONMENTAL DATA
dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-environmental/'

temp_in = pandas.DataFrame(columns=cols) # inside temperature
temp_out = pandas.DataFrame(columns=cols) # outside temperature
hum_in = pandas.DataFrame(columns=cols) # inside humidity
hum_out = pandas.DataFrame(columns=cols) # outside humidity

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
        env_raw =pandas.read_csv(r'C:\Users\SABA\Google Drive\mtsg\data\homeB-all\homeB-power\2012-Apr-15.csv',header=None,sep=",",usecols=[0,1,2,3,4], names=['timestamp','temp_in','temp_out','hum_in','hum_out'],index_col=[0]) # load loads
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=env_raw.index.min() # first value of new index
		idx=pandas.Index(numpy.arange(start=start,stop=start+60*60*24,step=300)) # timestamps for the whole day, end at 23:59:59
		env_full=env_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		temp_in_hrs=load_full.as_matrix(columns=['temp_in']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging inside temperature
		temp_out_hrs=load_full.as_matrix(columns=['temp_out']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging outside temperature
		hum_in_hrs=load_full.as_matrix(columns=['hum_in']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging inside humidity
		hum_out_hrs=load_full.as_matrix(columns=['hum_out']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging outside humidity
		temp_in.loc[temp_in.shape[0]]=temp_in_hrs
		temp_out.loc[temp_out.shape[0]]=temp_out_hrs
		hum_in.loc[hum_in.shape[0]]=hum_in_hrs
		hum_out.loc[hum_out.shape[0]]=hum_out_hrs



mp.plot(loads.loc[80])
mp.show()

# MISSING DATA HISTOGRAM 

dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
#loads_all = pandas.DataFrame(columns=['time','load'])
loads_list=[]

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pandas.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, names=['time','load'],index_col=[0]) # load loads
		loads_list.append(load_raw)
		
loads_all=pandas.concat(loads_list)		





# LOADING LOAD DATA
dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
loads = pandas.DataFrame()

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pandas.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, date_parser=(lambda x:pandas.to_datetime(x,unit='s')), names=['time','load'],index_col=[0]) # load loads
		if load_raw.shape[0]/(60*60*24)<0.90: # discard file with more than 5% missing data
			continue
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=load_raw.index.min() # first value of new index
		idx=pandas.date_range(start=start,periods=86400,freq='1S') # timestamps for the whole day, end at 23:59:59
		load_full=load_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		load_hrs=load_full.resample('H').sum() # hourly aggregation of loads
		if (loads.shape[0]<10):
			load_raw.plot()
			load_hrs.plot()
		# build new entry for current day
		load_hrs['date']=pandas.DatetimeIndex(load_hrs.index).date # new index
		load_hrs['time']=pandas.DatetimeIndex(load_hrs.index).time # new columns
		loads=loads.append(load_hrs.pivot(index='date', columns='time', values='load')) # pivot & append new entry

loads.ix['2012-06-04'].plot(y='load')

mp.plot(numpy.arange(24),loads.ix[19].values[1:])
mp.plot(numpy.arange(24),load_hrs.tolist())
mp.show() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
loads = pandas.DataFrame(columns=['day'] + cols)

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pandas.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, date_parser=(lambda x:pandas.to_datetime(x,unit='s')), names=['time','load'],index_col=[0]) # load loads
		if load_raw.shape[0]/(60*60*24)<0.90: # discard file with more than 5% missing data
			continue
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=load_raw.index.min() # first value of new index
		idx=pandas.date_range(start=start,periods=86400,freq='1S') # timestamps for the whole day, end at 23:59:59
		#idx=pandas.Index(numpy.arange(start,start+60*60*24)) # timestamps for the whole day, end at 23:59:59
		load_full=load_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		load_full.resample('H').sum() # aggregate hours
		#load_hrs=load_full.as_matrix(columns=['load']).reshape(-1,load_full.shape[0]//time_ints).sum(axis=1) # aggregation for time intervals
		pandas.concat(loads,)
		loads.loc[loads.shape[0]]=[start]+load_hrs.tolist()

loads.set_index('day', inplace=True)

 
 
 
 
 
load_dh=numpy.reshape(load_h,(365,-1),order='C')

X=numpy.arange(0,numpy.size(load_dh, axis=0))


seed=0
np.random.seed(seed)


#create model
print('creating model')

model = Sequential()
model.add(Dense(100, input_dim=1, init='uniform', activation='relu'))
model.add(Dense(24, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# training
print('Training')

model.fit(X, load_dh, batch_size=10, nb_epoch=10000, verbose=2, validation_split=0.3, shuffle=True)

scores = model.evaluate(X, load_dh)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
