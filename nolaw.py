'''
A Recursive Non-Linear Adaptive Wiener Filter for Time Series Denoising

Python implementation for the recursive NoLAW filter

Author: Alexandre L. M. Levada

'''

import sys
import math
import warnings
import pandas
import numpy as np
import pandas as pd
import sklearn.datasets as skdata
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from locale import atof
from sklearn import metrics
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from darts import TimeSeries
from darts.models import KalmanFilter
from darts.models import GaussianProcessFilter
from sklearn.gaussian_process.kernels import RBF

# To avoid some warnings
warnings.simplefilter('ignore')

# Binomial of n over k (to generate the Gaussian filter)
def binomial(n, k):
	return (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))

# Gaussian filter
def gaussian_filter(n):
	h = np.zeros(n+1)
	for i in range(n+1):
		h[i] = (binomial(n,i))
	return h/sum(h)

# Mean filter
def mean_filter(n):
	h = (1/n)*np.ones(n) 
	return h

'''
Linear Wiener filter 

Parameters
signal: the input signal
k: controls the size of the local windows (neighborhood order)
	   size = 2*k + 1
       1 -> 3 samples
       2 -> 5 samples
       3 -> 7 samples
noise_var: the noise variance
'''
def linear_wiener_filter(signal, k, noise_var):
	# Output signal
	restored = np.zeros(len(signal))
	# Boundary value problem
	signal = np.pad(signal, k, 'reflect')
	# Application of the filter
	for i in range(k, len(signal)-k):
		# Local average and variance
		mean = np.mean(signal[i-k:i+k+1])
		variance = np.var(signal[i-k:i+k+1])
		# To avoid negative variance
		var_s = variance - noise_var
		if var_s < 0:
			var_s = 0
		# Filter the i-th sample 
		restored[i-k] = mean + (var_s/variance)*(signal[i] - mean)
		if np.isnan(restored[i-k]):
			restored[i-k] = signal[i]
	return restored

'''
NoLAW filter

Parameters:
signal: the input signal
order: the order of the filter (quadratic, cubic etc.)
k: controls the size of the local windows (neighborhood order)
	   size = 2*k + 1
       1 -> 3 samples
       2 -> 5 samples
       3 -> 7 samples
noise_var: the noise variance
'''
def NoLAW_filter(sinal, order, k, noise_var):
	# If k = 1, perform the linear Wiener filtering
	if order == 1:
		suave = linear_wiener_filter(sinal, k, noise_var)
		return suave
	else:
		# Array to store the Restored signal
		restored = np.zeros(len(sinal))
		suave = NoLAW_filter(sinal, order-1, k, noise_var)		
		# Boundary value problem
		suave = np.pad(suave, k, 'reflect')
		sinal = np.pad(sinal, k, 'reflect')
		for i in range(k, len(sinal)-k):
			num_M = 2*order + 1 			# Number of distinct terms in matrix M
			# Local statistics
			vector_M = np.zeros(num_M)
			for j in range(num_M):
				vector_M[j] = np.mean(sinal[i-k:i+k+1]**j)
			# Build the matrix M
			dim_M = order + 1
			M = np.zeros((dim_M, dim_M))
			for j in range(dim_M):
				for l in range(dim_M):
					M[j, l] = vector_M[-1-l-j]
			# Regularization if matrix M is ill-conditioned
			if np.linalg.cond(M) > 1/sys.float_info.epsilon:
				M = M + np.diag(0.0001*np.ones(dim_M))
			# Build the vector C
			c = np.zeros(dim_M)
			for j in range(dim_M):
				exp = (dim_M - 1) - j
				c[j] = np.mean((sinal[i-k:i+k+1]**(exp)*suave[i-k:i+k+1]))
			# Solve the linear system M x = C
			sol = np.linalg.solve(M, c)
			# Filter the i-th sample
			for j in range(len(sol)):
				restored[i-k] += sol[j]*sinal[i]**(order - j)
			if np.isnan(restored[i-k]):
				restored[i-k] = suave[i]
		return restored

'''
Estimation of the noise variance

Jakub Pęksiński, Grzegorz Mikołajczak, Estimation of the noise variance in time series using
a median filter Computer Applications in Electrical Engineering, Vol. 12, 2014.

'''
def estimate_noise_variance(signal, window=5):
	result = median_filter(signal, size=window)
	V0 = np.var(signal)
	Vmed = np.var(result)
	noise_var = (2*len(signal)/(2*len(signal) - np.pi))*(V0 - Vmed)
	return noise_var

# Mean Absolute Percentage Error
def MAPE(y_true, y_pred):
	for i in range(len(y_true)):
		if y_true[i] == 0:
			y_true[i] = 0.1
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Peak Signal to Noise ratio
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    MAX = original.max()
    PSNR = 20 * math.log10(MAX / math.sqrt(mse))
    return PSNR


###############################
# Loading data
###############################
# .txt files
filename = 'bovespa.txt'
# filename = 'dolar.txt'
# filename = 'CO2_Canada.txt'
# filename = 'shampoo.txt'
# filename = 'wind.txt'
# filename = 'gasprice.txt'
# filename = 'sp500.txt'
# filename = 'tundra.txt'
# filename = 'ThaiDrug.txt'
# filename = 'Beaver.txt'
# filename = 'Cardio.txt'
# filename = 'beer.txt'
# filename = 'Baltimore.txt'
# .csv files
# filename = 'Cali.csv'
# filename = 'monthly-housing.csv'
# filename = 'ETTh1.csv'
# filename = 'US annual bond.csv'
# filename = 'IBM.csv'
# filename = 'Interest rates.csv'
# filename = 'Forces.csv'

# Reading data from file
if filename[-3:] == 'txt':
	signal = np.loadtxt(filename)
else:
	signal = pandas.read_csv(filename)			

# if the time series is a pandas dataframe, we need to convert it to a numpy array
if type(signal) == pandas.core.frame.DataFrame:
	signal = signal.iloc[:, -1].to_numpy().astype(float)

noise_std = np.var(signal)**0.5
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.05*noise_std, len(signal))		# 1% of the signal's standard deviation

# Add noise to time series
signal_n = signal + noise

# Number of samples
n = len(signal)

# Gaussian filter size
# 1 -> 3 samples
# 2 -> 5 samples
# 3 -> 7 samples
k = 1
# Number of samples in the Gaussian filter
dim = 2*k + 1

# Gaussian filter
h = gaussian_filter(dim)
gaussian_smooth = np.convolve(np.pad(signal_n, k, 'reflect'), h, mode='same')
gaussian_smooth = gaussian_smooth[k:len(gaussian_smooth)-k]

# Holt's method - Triple Exponential Smoothing
fit2 = Holt(signal_n, initialization_method='estimated').fit()
holt_smooth = fit2.fittedvalues

noise_var = estimate_noise_variance(signal_n)
print('Estimated noise variance: %f' %noise_var)
print()

# Linear Wiener filter
wiener1 = linear_wiener_filter(signal_n, 1, noise_var)		# k = 1
wiener2 = linear_wiener_filter(signal_n, 2, noise_var)		# k = 2
wiener3 = linear_wiener_filter(signal_n, 3, noise_var)		# k = 3

# NoLAW filter
nolaw_f_order_3_1 = NoLAW_filter(signal_n, 3, 1, noise_var)		# order = 3, k = 1
nolaw_f_order_5_1 = NoLAW_filter(signal_n, 5, 1, noise_var)		# order = 5, k = 1
nolaw_f_order_3_2 = NoLAW_filter(signal_n, 3, 2, noise_var)		# order = 3, k = 2
nolaw_f_order_5_2 = NoLAW_filter(signal_n, 5, 2, noise_var)		# order = 5, k = 2
nolaw_f_order_3_3 = NoLAW_filter(signal_n, 3, 3, noise_var)		# order = 3, k = 3
nolaw_f_order_5_3 = NoLAW_filter(signal_n, 5, 3, noise_var)		# order = 5, k = 3

####################################
# Performance evaluation metrics
####################################
mape_g = MAPE(signal, gaussian_smooth)
print('MAPE Gaussian: %f' %mape_g)
mape_h = MAPE(signal, holt_smooth)
print('MAPE Holt: %f' %mape_h)
mape_w1 = MAPE(signal, wiener1)
print('MAPE Wiener (ordem 1): %f' %mape_w1)
mape_nolaw_3_1 = MAPE(signal, nolaw_f_order_3_1)
print('MAPE NoLAW grau 3 (ordem 1): %f' %mape_nolaw_3_1)
mape_nolaw_5_1 = MAPE(signal, nolaw_f_order_5_1)
print('MAPE NoLAW grau 5 (ordem 1): %f' %mape_nolaw_5_1)
mape_w2 = MAPE(signal, wiener2)
print('MAPE Wiener (ordem 2): %f' %mape_w2)
mape_nolaw_3_2 = MAPE(signal, nolaw_f_order_3_2)
print('MAPE NoLAW grau 3 (ordem 2): %f' %mape_nolaw_3_2)
mape_nolaw_5_2 = MAPE(signal, nolaw_f_order_5_2)
print('MAPE NoLAW grau 5 (ordem 2): %f' %mape_nolaw_5_2)
mape_w3 = MAPE(signal, wiener3)
print('MAPE Wiener (ordem 3): %f' %mape_w3)
mape_nolaw_3_3 = MAPE(signal, nolaw_f_order_3_3)
print('MAPE NoLAW grau 3 (ordem 3): %f' %mape_nolaw_3_3)
mape_nolaw_5_3 = MAPE(signal, nolaw_f_order_5_3)
print('MAPE NoLAW grau 5 (ordem 3): %f' %mape_nolaw_5_3)
print()

mapes = np.array([mape_g, mape_h, mape_w1, mape_nolaw_3_1, mape_nolaw_5_1, mape_w2, mape_nolaw_3_2, mape_nolaw_5_2, mape_w3, mape_nolaw_3_3, mape_nolaw_5_3])
np.savetxt('MAPE.txt', mapes, fmt='%.4f', delimiter=',', newline='\t')

print('MAE Gaussian: %f' %(metrics.median_absolute_error(signal, gaussian_smooth)))
print('MAE Holt: %f' %(metrics.median_absolute_error(signal, holt_smooth)))
print('MAE Wiener (order 1): %f' %(metrics.median_absolute_error(signal, wiener1)))
print('MAE NoLAW grau 3 (ordem 1): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_3_1)))
print('MAE NoLAW grau 5 (ordem 1): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_5_1)))
print('MAE Wiener (order 2): %f' %(metrics.median_absolute_error(signal, wiener2)))
print('MAE NoLAW grau 3 (ordem 2): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_3_2)))
print('MAE NoLAW grau 5 (ordem 2): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_5_2)))
print('MAE Wiener (order 3): %f' %(metrics.median_absolute_error(signal, wiener3)))
print('MAE NoLAW grau 3 (ordem 3): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_3_3)))
print('MAE NoLAW grau 5 (ordem 3): %f' %(metrics.median_absolute_error(signal, nolaw_f_order_5_3)))
print()

print('PSNR Gaussian: %f' %(psnr(signal, gaussian_smooth)))
print('PSNR Holt: %f' %(psnr(signal, holt_smooth)))
print('PSNR Wiener (order 1): %f' %(psnr(signal, wiener1)))
print('PSNR NoLAW grau 3 (ordem 1): %f' %(psnr(signal, nolaw_f_order_3_1)))
print('PSNR NoLAW grau 5 (ordem 1): %f' %(psnr(signal, nolaw_f_order_5_1)))
print('PSNR Wiener (order 2): %f' %(psnr(signal, wiener2)))
print('PSNR NoLAW grau 3 (ordem 2): %f' %(psnr(signal, nolaw_f_order_3_2)))
print('PSNR NoLAW grau 5 (ordem 2): %f' %(psnr(signal, nolaw_f_order_5_2)))
print('PSNR Wiener (order 3): %f' %(psnr(signal, wiener3)))
print('PSNR NoLAW grau 3 (ordem 3): %f' %(psnr(signal, nolaw_f_order_3_3)))
print('PSNR NoLAW grau 5 (ordem 3): %f' %(psnr(signal, nolaw_f_order_5_3)))
print()


# ##################################
# # Plot figures
# ##################################
plt.figure(1)
plt.plot(signal, color='blue', label='Original')
plt.plot(signal_n, color='red', label='Noisy')
plt.xlabel('time')
plt.ylabel('values')
plt.title(filename[:-4] + ' time series')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(signal, color='green', label='Original')
plt.plot(wiener2, color='red', label='Wiener')
plt.plot(nolaw_f_order_5_2, color='blue', label='NoLAW')
plt.xlabel('time')
plt.ylabel('values')
plt.title('Bovespa time series')
plt.legend()
plt.show()