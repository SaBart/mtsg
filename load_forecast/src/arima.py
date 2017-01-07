'''ARIMA'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA



def data_stats(data,window=7):
	from statsmodels.tsa.stattools import adfuller
	#Determing rolling statistics
	mean_rol=data.rolling(window=window).mean()
	std_rol=data.rolling(window=window).std()
	#Plot rolling statistics:
	orig = plt.plot(data, color='blue',label='load')
	mean = plt.plot(mean_rol, color='red', label='rolling mean')
	std = plt.plot(std_rol, color='green', label = 'rolling std')
	plt.legend(loc='best')
	plt.title('rolling Mean & standard Deviation')
	plt.show(block=False)
	#Perform Dickey-Fuller test:
	print('Results of Dickey-Fuller Test:')
	dftest = adfuller(data, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)
	
# perform Dickey-Fuller test to check data stacionarity:	
def df_test(data):
	from statsmodels.tsa.stattools import adfuller	
	print('Results of Dickey-Fuller Test:')
	df_test = adfuller(data, autolag='AIC')
	dfoutput = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in df_test[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)
	
	
# perform Dickey-Fuller test for each hour	
for i in range(0,24):
	print('{}:'.format(i))
	df_test(load_with_nans[i].dropna())
	print()
	
	
data=pd.DataFrame(load_with_nans[11].fillna(method='bfill'))
data_dec=seasonal_decompose(data.values,freq=7)
data_dec.plot()
	
lag_acf = acf(data, nlags=50)
lag_pacf = pacf(data, nlags=50, method='ols')



fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(2,1,1)
fig = plot_acf(data_dec.resid,lags=50,ax=ax1) # plot ACF
ax2 = fig.add_subplot(2,1,2)
plot_pacf(data_dec.resid,lags=50,ax=ax2) # plot PACF
	
acf,ci,Q,p_value = acf(data, nlags=50,alpha=0.05,  qstat=True, unbiased=True)

# AR model
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-data)**2))

# MA model
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-data)**2))

# AR+MA model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-data)**2))






#Plot ACF: 
plt.subplot(1,2,1) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(1,2,2)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

