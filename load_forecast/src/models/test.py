'''
Created on 4 Dec 2016

@author: SABA
'''

import pandas
import numpy
import sklearn
import matplotlib.pyplot as mp

from keras.models import Sequential
from keras.layers.core import Dense

# loading load data

load_raw =pandas.read_csv(r'C:\Users\SABA\Google Drive\mtsg\data\homeB-all\homeB-power\2012-Apr-15.csv',header=None,sep=",",usecols=[0,1], names=['timestamp','load'],index_col=[0]) # load loads

# add previous values for missing timestamps
# maybe predict them from a couple of previus values?
start=load_raw.index.min() # first value of new index
idx=pandas.Index(numpy.arange(start,start+60*60*24)) # timestamps for the whole day, end at 23:59:59
load_filled=load_raw.reindex(idx, method='nearest') # fill missing with previous values

# hourly aggregation

load_hours=load_filled.as_matrix(columns=['load']).reshape(-1,3600).sum(axis=1)

mp.plot(numpy.arange(1,25),load_hours)
mp.show()

 
#load_dh=numpy.reshape(load_h,(365,-1),order='C')

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

