'''
Created on 4 Dec 2016

@author: SABA
'''

import os.path
import sys
import csv
import numpy as np
import sklearn as skl
import matplotlib.pyplot as mp

from keras.models import Sequential
from keras.layers.core import Dense

if __name__ == '__main__':
	
	seed=0
	np.random.seed(seed)

	# loading data
	print('Loading data')
	
	load_csv=r'C:\Users\SABA\Google Drive\mtsg\code\load_forecast\data\load.csv'
	
	
	with open(load_csv, 'r') as csvfile:
		load_raw=np.array(list(csv.reader(csvfile))).astype('float')
		
	load_h=np.reshape(load_raw,(-1,60),order='C').sum(axis=1) # hourly aggregation
	load_dh=np.reshape(load_h,(365,-1),order='C')
	
	X=np.arange(0,np.size(load_dh, axis=0))
	
	
	#mp.plot(np.arange(1,25),load_dh[1])
	#mp.show()
	
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
	
	