'''
Created on 4 Dec 2016

@author: SABA
'''
import os
import numpy
import sklearn
import pandas
import matplotlib.pyplot as mp
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
		# build new entry for current day
		load_hrs['date']=pandas.DatetimeIndex(load_hrs.index).date # new index
		load_hrs['time']=pandas.DatetimeIndex(load_hrs.index).time # new columns
		loads=loads.append(load_hrs.pivot(index='date', columns='time', values='load')) # pivot & append new entry

loads.ix['2012-06-04'].plot(y='load')

mp.plot(numpy.arange(24),loads.ix[19].values[1:])
mp.plot(numpy.arange(24),load_hrs.tolist())
mp.show() 



# loads generator data
def load_data(path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv'):
	load_raw =pandas.read_csv(path,header=None,sep=",",usecols=[0], names=['load'],dtype={'load': numpy.float64}) # load loads
	load=load_raw.groupby(load_raw.index//60).sum() # hourly aggregation
	load['hour']=pandas.Series(numpy.concatenate([numpy.arange(1,25)]*365)) # new column for pivoting
	load['day']=pandas.Series(numpy.repeat(numpy.arange(1,366), repeats=24)) # new column for pivoting
	load=load.pivot(index='day',columns='hour',values='load') # pivoting
	return load

# lags data
def lag_data(data,lag=1):
	data_lagged={} # lagged dataframes for merging
	for i in range(0,lag+1): # for each time step
		data_lagged[i-lag]=data.shift(-i) # add lagged dataframe
	res=pandas.concat(data_lagged.values(),axis=1,join='inner',keys=data_lagged.keys()) # merge lagged dataframes	
	return res.dropna()

# separates data into training & testing sets & converts dataframes to numpy matrices 
def format_data(path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv', lag=1, test_size=0.2):
	data=lag_data(load_data(path),lag)
	train, test =split_train_test(data, test_size)
	X_train,Y_train=split_X_Y(train)
	X_test,Y_test=split_X_Y(test)
	return X_train.as_matrix(), Y_train.as_matrix(), X_test.as_matrix(), Y_test.as_matrix()

# split data into X & Y
def split_X_Y(data):
	X=data.select(lambda x:x[0] not in [0], axis=1)
	Y=data[0]
	return X, Y

# split data into train & test sets
def split_train_test(data, test_size=0.2):
	from sklearn.model_selection import train_test_split
	train, test =train_test_split(data, test_size=test_size)
	return train,test
	
# create & train basic NN model
def create_model(nb_in=24, nb_out=24, nb_hidden=50, nb_epoch=200, batch_size=1, activation='relu', loss='mean_squared_error', optimizer='adam'):
	from keras.models import Sequential
	from keras.layers.core import Dense
	model = Sequential() # FFN
	model.add(Dense(nb_hidden, input_dim=nb_in,activation=activation)) # input & hidden layers
	#model.add(Dropout({{uniform(0, 1)}})) # randomly set a number of inputs to 0 to prevent overfitting
	model.add(Dense(nb_out)) # output layer
	model.compile(loss=loss, optimizer=optimizer) # assemble network	
	return model

from sklearn.metrics import mean_squared_error, make_scorer
mse = make_scorer(mean_squared_error, multioutput='uniform_average')

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

seed=0 # fix seed for reprodicibility
numpy.random.seed(seed)
path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv' # data path
X,Y=split_X_Y(lag_data(load_data(path),lag=1)) # prepare data
model = KerasClassifier(build_fn=create_model)

#nb_hidden=[10,20,30,40,50,60,70,80,90,100] # domain for number of hidden neurons
nb_in=[X.shape[1]]
nb_out=[Y.shape[1]]
nb_hidden=[10,20,30]
param_grid={'nb_hidden':nb_hidden} # grid parameter space
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, scoring='r2')
grid_result = grid.fit(X.as_matrix(), Y.as_matrix())
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




model=create_model(X_train.shape[1],Y_train.shape[1])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
data_pred=model.predict(X_test)

mp.plot(data_pred[0])
mp.plot(Y_test[0])
mp.show()



# Use scikit-learn to grid search the number of neurons
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=8, init='uniform', activation='linear', W_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/src/models/pima-indians-diabetes.csv', delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim


for l in range(1,3):
	best_run, best_model = optim.minimize(model=model,
										  data=data(l),
										  algo=tpe.suggest,
										  max_evals=5,
										  trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))







	









def cut_data(data,steps=1):
	new_dict={}
	tuples=[] # multiIndex
	for col in data:
		new_dict[col]=data[col] # add old column
		for i in range(1,steps+1):
			new_dict['%s_lag%d' %(col,i)]=data[col].shift(i) # add shifted column
			tuples.append((i,col)) # new multiIndex entry 
	res=pandas.DataFrame(new_dict,index=data.index) 
	return res.dropna(axis=0)
	
	
	
	
df1 = pandas.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']}, index=[0, 1, 2, 3])

df2 = pandas.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],'B': ['B4', 'B5', 'B6', 'B7'],'C': ['C4', 'C5', 'C6', 'C7'],'D': ['D4', 'D5', 'D6', 'D7']},index=[4, 5, 6, 7])





	
	for col in data:
		new_dict[col]=data[col] # add old column
		for i in range(1,steps+1):
			new_dict['%s_lag%d' %(col,i)]=data[col].shift(i) # add shifted column 
	res=pandas.DataFrame(new_dict,index=data.index) 
	return res.dropna(axis=0)




load.plot(kind='bar')




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



# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = pandas.read_csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/src/models/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# reshape dataset
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
mp.plot(dataset)
mp.plot(trainPredictPlot)
mp.plot(testPredictPlot)
mp.show()
