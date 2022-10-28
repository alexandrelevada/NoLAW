'''
Non-linear Adaptive Wiener Filter for Time Series Smoothing 

Python implementation for the NoLAW filter

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
from scipy.ndimage import median_filter
from locale import atof
from sklearn import metrics
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# To avoid some warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

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
sinal: the input signal
ordem: controls the size of the local windows (neighborhood order)
       1 -> 3 samples
       2 -> 5 samples
       3 -> 7 samples
var_ruido: the noise variance
'''
def linear_wiener_filter(sinal, ordem, var_ruido):
	# Output signal
	restored = np.zeros(len(sinal))
	# Boundary value problem
	sinal = np.pad(sinal, ordem, 'reflect')
	# Application of the filter
	for i in range(ordem, len(sinal)-ordem):
		# Local average and variance
		media = np.mean(sinal[i-ordem:i+ordem+1])
		variancia = np.var(sinal[i-ordem:i+ordem+1])
		# To avoid negative variance
		var_f = variancia - var_ruido
		if var_f < 0:
			var_f = 0
		# Filter the i-th sample 
		restored[i-ordem] = media + (var_f/variancia)*(sinal[i] - media)
		if np.isnan(restored[i-ordem]):
			restored[i-ordem] = sinal[i]
	return restored

'''
Quadratic Wiener filter 

Parameters
sinal: the input signal
ordem: controls the size of the local windows (neighborhood order)
       1 -> 3 samples
       2 -> 5 samples
       3 -> 7 samples
var_ruido: the noise variance
'''
def quadratic_wiener_filter(sinal, ordem, var_ruido):
	# Output signal
	restored = np.zeros(len(sinal))
	# Boundary value problem
	sinal = np.pad(sinal, ordem, 'reflect')
	# Pre-smoothed version of the signal
	suave = linear_wiener_filter(sinal, ordem, var_ruido)
	# Application of the filter
	for i in range(ordem, len(sinal)-ordem):
		# Local statistics
		M4 = np.mean(sinal[i-ordem:i+ordem+1]**4)
		M3 = np.mean(sinal[i-ordem:i+ordem+1]**3)
		M2 = np.mean(sinal[i-ordem:i+ordem+1]**2)
		media = np.mean(sinal[i-ordem:i+ordem+1])
		variancia = np.var(sinal[i-ordem:i+ordem+1])
		# To avoid negative variance
		var_f = variancia - var_ruido
		if var_f < 0:
			var_f = 0
		# Define the system of equations
		A = np.zeros((3, 3))
		b = np.zeros(3)
		A[0, 0] = M4
		A[0, 1] = M3
		A[0, 2] = M2
		A[1, 0] = M3
		A[1, 1] = M2
		A[1, 2] = media
		A[2, 0] = M2
		A[2, 1] = media
		A[2, 2] = 1
		# Small regularization
		if np.linalg.cond(A) > 1/sys.float_info.epsilon:
			A = A + np.diag(0.001*np.ones(3))
		b[0] = np.mean((sinal[i-ordem:i+ordem+1]**2)*suave[i-ordem:i+ordem+1])
		b[1] = np.mean(sinal[i-ordem:i+ordem+1]*suave[i-ordem:i+ordem+1])
		b[2] = np.mean(suave[i-ordem:i+ordem+1])
		# Solve the system
		sol = np.linalg.solve(A, b)
		# Filter the i-th sample
		restored[i-ordem] = sol[0]*sinal[i]**2 + sol[1]*sinal[i] + sol[2]
		if np.isnan(restored[i-ordem]):
			restored[i-ordem] = suave[i]
	return restored


'''
Cubic Wiener filter 

Parameters
sinal: the input signal
ordem: controls the size of the local windows (neighborhood order)
       1 -> 3 samples
       2 -> 5 samples
       3 -> 7 samples
var_ruido: the noise variance
'''
def cubic_wiener_filter(sinal, ordem, var_ruido):
	# Output signal
	restored = np.zeros(len(sinal))
	# Boundary value problem
	sinal = np.pad(sinal, ordem, 'reflect')
	# Pre-smoothed version of the signal
	suave = quadratic_wiener_filter(sinal, ordem, var_ruido)
	# Application of the filter
	for i in range(ordem, len(sinal)-ordem):
		# Local statistics
		M6 = np.mean(sinal[i-ordem:i+ordem+1]**6)
		M5 = np.mean(sinal[i-ordem:i+ordem+1]**5)
		M4 = np.mean(sinal[i-ordem:i+ordem+1]**4)
		M3 = np.mean(sinal[i-ordem:i+ordem+1]**3)
		M2 = np.mean(sinal[i-ordem:i+ordem+1]**2)
		media = np.mean(sinal[i-ordem:i+ordem+1])
		variancia = np.var(sinal[i-ordem:i+ordem+1])
		mediana = np.median(sinal[i-ordem:i+ordem+1])
		# To avoid negative variances
		var_f = variancia - var_ruido
		if var_f < 0:
			var_f = 0
		# Define the system of equations
		A = np.zeros((4, 4))
		b = np.zeros(4)
		A[0, 0] = M6
		A[0, 1] = M5
		A[0, 2] = M4
		A[0, 3] = M3
		A[1, 0] = M5
		A[1, 1] = M4
		A[1, 2] = M3
		A[1, 3] = M2
		A[2, 0] = M4
		A[2, 1] = M3
		A[2, 2] = M2
		A[2, 3] = media
		A[3, 0] = M3
		A[3, 1] = M2
		A[3, 2] = media
		A[3, 3] = 1
		# Small regularization
		if np.linalg.cond(A) > 1/sys.float_info.epsilon:
			A = A + np.diag(0.001*np.ones(4))
		b[0] = np.mean((sinal[i-ordem:i+ordem+1]**3)*suave[i-ordem:i+ordem+1])
		b[1] = np.mean((sinal[i-ordem:i+ordem+1]**2)*suave[i-ordem:i+ordem+1])
		b[2] = np.mean(sinal[i-ordem:i+ordem+1]*suave[i-ordem:i+ordem+1])
		b[3] = np.mean(suave[i-ordem:i+ordem+1])
		# Solve the system
		sol = np.linalg.solve(A, b)
		# Filter the i-th sample
		restored[i-ordem] = sol[0]*sinal[i]**3 + sol[1]*sinal[i]**2 + sol[2]*sinal[i] + sol[3]
		if np.isnan(restored[i-ordem]):
			restored[i-ordem] = suave[i]
	return restored

'''
Estimation of the noise variance

Jakub Pęksiński, Grzegorz Mikołajczak, Estimation of the noise variance in time series using
a median filter Computer Applications in Electrical Engineering, Vol. 12, 2014.

'''
def estimate_noise_variance(sinal, janela=5):
	result = median_filter(sinal, size=janela)
	V0 = np.var(sinal)
	Vmed = np.var(result)
	var_ruido = (2*len(sinal)/(2*len(sinal) - np.pi))*(V0 - Vmed)
	return var_ruido

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
#sinal = np.loadtxt('../daily_female_births.txt')			
#sinal = np.loadtxt('../bovespa.txt')					
#sinal = np.loadtxt('../dolar.txt')					
#sinal = np.loadtxt('../CO2_Canada.txt')				
#sinal = np.loadtxt('../seatbelts.txt')					
#sinal = np.loadtxt('../beer.txt')						
sinal = np.loadtxt('../MinTemp.txt')					
#sinal = np.loadtxt('../shampoo.txt')					

# .cvs files
#data = pandas.read_csv('../Cali Emissions.csv')			
#data = pandas.read_csv('../Nuclear Capacity.csv')			
#data = pandas.read_csv('../monthly-sunspots.csv')			
#data = pandas.read_csv('../ETTh1.csv')		 # Electricity Transformer Dataset: last column (index -1) 
#data = pandas.read_csv('../ETTh2.csv')		 # Electricity Transformer Dataset: first column (index 1) 

# if the time series is a pandas dataframe, we need to convert it to a numpy array
#sinal = data.iloc[:, -1].to_numpy()



# Gaussian Noise generation
var_sinal = sinal.var()
ruido = np.random.normal(0, np.sqrt(0.1*var_sinal), len(sinal))		# 10% of the signal's variance
sinal_r = sinal + ruido

n = len(sinal)

# Order of the neighborhood sysmtem
# 2 -> 5 samples
ordem = 2
# Number of samples in the local windows
dim = 2*ordem + 1

# Gaussian filter
h = gaussian_filter(dim)
gaussian_smooth = np.convolve(sinal_r, h, mode='same')

# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(sinal_r, initialization_method="estimated").fit()
ses_smooth = fit1.fittedvalues

# Holt's method - Triple Exponential Smoothing
fit2 = Holt(sinal_r, initialization_method="estimated").fit()
holt_smooth = fit2.fittedvalues

###### NoLAW window of size 3
# Estimate noise variance
var_ruido = estimate_noise_variance(sinal_r, 3)
print('Noise variance (window of size 3): %f' %var_ruido)
print()

wiener_smooth_1 = linear_wiener_filter(sinal_r, ordem-1, var_ruido)
quadratic_wiener_smooth_1 = quadratic_wiener_filter(sinal_r, ordem-1, var_ruido)
cubic_wiener_smooth_1 = cubic_wiener_filter(sinal_r, ordem-1, var_ruido)

##### NoLAW window of size 5
# Estimate noise variance
var_ruido = estimate_noise_variance(sinal_r)
print('Noise variance (window of size 5): %f' %var_ruido)
print()

wiener_smooth_2 = linear_wiener_filter(sinal_r, ordem, var_ruido)
quadratic_wiener_smooth_2 = quadratic_wiener_filter(sinal_r, ordem, var_ruido)
cubic_wiener_smooth_2 = cubic_wiener_filter(sinal_r, ordem, var_ruido)

##### NoLAW window of size 7
# Estimate noise variance
var_ruido = estimate_noise_variance(sinal_r, 7)
print('Noise variance (window of size 7): %f' %var_ruido)
print()

wiener_smooth_3 = linear_wiener_filter(sinal_r, ordem+1, var_ruido)
quadratic_wiener_smooth_3 = quadratic_wiener_filter(sinal_r, ordem+1, var_ruido)
cubic_wiener_smooth_3 = cubic_wiener_filter(sinal_r, ordem+1, var_ruido)

####################################
# Performance evaluation metrics
####################################
print('MAPE Gaussian: %f' %(MAPE(sinal, gaussian_smooth)))
print('MAPE SES: %f' %(MAPE(sinal, ses_smooth)))
print('MAPE Holt: %f' %(MAPE(sinal, holt_smooth)))
print('MAPE linear Wiener (window 3): %f' %(MAPE(sinal, wiener_smooth_1)))
print('MAPE linear Wiener (window 5): %f' %(MAPE(sinal, wiener_smooth_2)))
print('MAPE linear Wiener (window 7): %f' %(MAPE(sinal, wiener_smooth_3)))
print('MAPE cubic Wiener (window 3): %f' %(MAPE(sinal, cubic_wiener_smooth_1)))
print('MAPE cubic Wiener (window 5): %f' %(MAPE(sinal, cubic_wiener_smooth_2)))
print('MAPE cubic Wiener (window 7): %f' %(MAPE(sinal, cubic_wiener_smooth_3)))
print()

print('MSE Gaussian: %f' %(metrics.mean_squared_error(sinal, gaussian_smooth)))
print('MSE SES: %f' %(metrics.mean_squared_error(sinal, ses_smooth)))
print('MSE Holt: %f' %(metrics.mean_squared_error(sinal, holt_smooth)))
print('MSE linear Wiener (window 3): %f' %(metrics.mean_squared_error(sinal, wiener_smooth_1)))
print('MSE linear Wiener (window 5): %f' %(metrics.mean_squared_error(sinal, wiener_smooth_2)))
print('MSE linear Wiener (window 7): %f' %(metrics.mean_squared_error(sinal, wiener_smooth_3)))
print('MSE cubic Wiener (window 3): %f' %(metrics.mean_squared_error(sinal, cubic_wiener_smooth_1)))
print('MSE cubic Wiener (window 5): %f' %(metrics.mean_squared_error(sinal, cubic_wiener_smooth_2)))
print('MSE cubic Wiener (window 7): %f' %(metrics.mean_squared_error(sinal, cubic_wiener_smooth_3)))
print()

print('MAE Gaussian: %f' %(metrics.mean_absolute_error(sinal, gaussian_smooth)))
print('MAE SES: %f' %(metrics.mean_absolute_error(sinal, ses_smooth)))
print('MAE Holt: %f' %(metrics.mean_absolute_error(sinal, holt_smooth)))
print('MAE linear Wiener (window 3): %f' %(metrics.mean_absolute_error(sinal, wiener_smooth_1)))
print('MAE linear Wiener (window 5): %f' %(metrics.mean_absolute_error(sinal, wiener_smooth_2)))
print('MAE linear Wiener (window 7): %f' %(metrics.mean_absolute_error(sinal, wiener_smooth_3)))
print('MAE cubic Wiener (window 3): %f' %(metrics.mean_absolute_error(sinal, cubic_wiener_smooth_1)))
print('MAE cubic Wiener (window 5): %f' %(metrics.mean_absolute_error(sinal, cubic_wiener_smooth_2)))
print('MAE cubic Wiener (window 7): %f' %(metrics.mean_absolute_error(sinal, cubic_wiener_smooth_3)))
print()

print('Median Absolute Error Gaussian: %f' %(metrics.median_absolute_error(sinal, gaussian_smooth)))
print('Median Absolute Error SES: %f' %(metrics.median_absolute_error(sinal, ses_smooth)))
print('Median Absolute Error Holt: %f' %(metrics.median_absolute_error(sinal, holt_smooth)))
print('Median Absolute Error linear Wiener (window 3): %f' %(metrics.median_absolute_error(sinal, wiener_smooth_1)))
print('Median Absolute Error linear Wiener (window 5): %f' %(metrics.median_absolute_error(sinal, wiener_smooth_2)))
print('Median Absolute Error linear Wiener (window 7): %f' %(metrics.median_absolute_error(sinal, wiener_smooth_3)))
print('Median Absolute Error cubic Wiener (window 3): %f' %(metrics.median_absolute_error(sinal, cubic_wiener_smooth_1)))
print('Median Absolute Error cubic Wiener (window 5): %f' %(metrics.median_absolute_error(sinal, cubic_wiener_smooth_2)))
print('Median Absolute Error cubic Wiener (window 7): %f' %(metrics.median_absolute_error(sinal, cubic_wiener_smooth_3)))
print()

print('PSNR Gaussian: %f' %(psnr(sinal, gaussian_smooth)))
print('PSNR SES: %f' %(psnr(sinal, ses_smooth)))
print('PSNR Holt: %f' %(psnr(sinal, holt_smooth)))
print('PSNR linear Wiener (window 3): %f' %(psnr(sinal, wiener_smooth_1)))
print('PSNR linear Wiener (window 5): %f' %(psnr(sinal, wiener_smooth_2)))
print('PSNR linear Wiener (window 7): %f' %(psnr(sinal, wiener_smooth_3)))
print('PSNR cubic Wiener (window 3): %f' %(psnr(sinal, cubic_wiener_smooth_1)))
print('PSNR cubic Wiener (window 5): %f' %(psnr(sinal, cubic_wiener_smooth_2)))
print('PSNR cubic Wiener (window 7): %f' %(psnr(sinal, cubic_wiener_smooth_3)))
print()


##################################
# Plot figures
##################################
df1 = pd.DataFrame(dict(
     original = sinal,
))
#fig1 = px.line(df1, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig1 = px.line(df1, width=800, height=600, title='Original').update_layout(showlegend=False)
#fig.show()
fig1.write_image('./graficos/original.png')

df2 = pd.DataFrame(dict(
     noisy = sinal_r,
))
#fig2 = px.line(df2, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig2 = px.line(df2, width=800, height=600, title='Noisy').update_layout(showlegend=False)
fig2.write_image('./graficos/noisy.png')

df3 = pd.DataFrame(dict(
     gaussian = gaussian_smooth,
))
#fig3 = px.line(df3, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig3 = px.line(df3, width=800, height=600, title='Gaussian filter').update_layout(showlegend=False)
fig3.write_image('./graficos/gaussian.png')

df4 = pd.DataFrame(dict(
     exponential = ses_smooth,
))
#fig4 = px.line(df4, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig4 = px.line(df4, width=800, height=600, title='Simple Exponential Smoothing').update_layout(showlegend=False)
fig4.write_image('./graficos/exponential.png')

df5 = pd.DataFrame(dict(
     holt = holt_smooth,
))
#fig5 = px.line(df5, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig5 = px.line(df5, width=800, height=600, title='Holt Winters method').update_layout(showlegend=False)
fig5.write_image('./graficos/holt.png')

df6 = pd.DataFrame(dict(
     wiener = wiener_smooth_1,
))
#fig6 = px.line(df6, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig6 = px.line(df6, width=800, height=600, title='Linear Wiener (first-order neighborhood)').update_layout(showlegend=False)
fig6.write_image('./graficos/linear_wiener_1.png')

df7 = pd.DataFrame(dict(
     wiener = wiener_smooth_2,
))
#fig7 = px.line(df7, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig7 = px.line(df7, width=800, height=600, title='Linear Wiener (second-order neighborhood)').update_layout(showlegend=False)
fig7.write_image('./graficos/linear_wiener_2.png')

df8 = pd.DataFrame(dict(
     wiener = wiener_smooth_3,
))
#fig8 = px.line(df8, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig8 = px.line(df8, width=800, height=600, title='Linear Wiener (third-order neighborhood)').update_layout(showlegend=False)
fig8.write_image('./graficos/linear_wiener_3.png')

df9 = pd.DataFrame(dict(
     cubic_wiener = cubic_wiener_smooth_1,
))
#fig9 = px.line(df9, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig9 = px.line(df9, width=800, height=600, title='Cubic NoLAW (first-order neighborhood)').update_layout(showlegend=False)
fig9.write_image('./graficos/cubic_wiener_1.png')

df10 = pd.DataFrame(dict(
     cubic_wiener = cubic_wiener_smooth_2,
))
#fig10 = px.line(df10, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig10 = px.line(df10, width=800, height=600, title='Cubic NoLAW (second-order neighborhood)').update_layout(showlegend=False)
fig10.write_image('./graficos/cubic_wiener_2.png')

df11 = pd.DataFrame(dict(
     cubic_wiener = cubic_wiener_smooth_3,
))
#fig11 = px.line(df11, width=800, height=600).update_layout(xaxis_title='samples', yaxis_title='time series')
fig11 = px.line(df11, width=800, height=600, title='Cubic NoLAW (third-order neighborhood)').update_layout(showlegend=False)
fig11.write_image('./graficos/cubic_wiener_3.png')