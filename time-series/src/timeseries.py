#!usr/bin/python

# ===============================================================================
#                           Read Data
# ===============================================================================

import pickle
import glob
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARMA

from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv

import statsmodels.api as sm
from scipy.stats import chisquare
from pandas.tools.plotting import lag_plot


import matplotlib.mlab as mlab


def read_pickle(path):
    """

    :param path: string, path to the pickle
    :return: object returned by pickle.load
    """
    f = open(path, 'rb') # opening pickle file, use ".pickle.2" if you are using Python 2
    obj = pickle.load(f) # loading the object
    f.close()
    return obj

# ===============================================================================
#                       Find the segment needed
# ===============================================================================
# type(obj)
# obj.keys()
# segments = list(obj.keys())[0:1000]
# for segment in segments:
# 	series = obj[segment] # get one element of obj
# 	print (segment)
#	print(series.count())
#	print(len(series))


# ===============================================================================
#                      Output pickle to CSV
# ===============================================================================
file_list = glob.glob("/scratch/sri223/segments/final/*.pickle")

chosen_segments = [('503100', '503970'), ('503970', '503842'), ('401897', '401898'),
                ('100652', '104002'), ('100195', '100197')]

for filepath in file_list:
    obj = read_pickle(filepath)
    for i in range(len(chosen_segments)):
        segment = chosen_segments[i]
        series = obj[segment]
        series.sort_index(inplace=True)
        series.fillna(method='ffill', inplace = True)
        with open('SEG' + str(i) + '.csv', 'a') as f:
            series.to_csv(f, header=False)

# ===============================================================================
#                      Plot the basic time series
# ===============================================================================


segment_id = 0
# filepath = '/home/yl5090/SEG' + str(i) + '.csv'
filepath = './SEG{}.csv'.format(str(segment_id))
df = read_csv(filepath, names=['Speed'],header=None, index_col=0)
df.fillna(method='ffill', inplace = True)

# Check distributions to detect outliers
import numpy as np
plt.hist(df.values, bins = 10)
plt.show()


# Remove outliers
def replace_outliers(data, m=3):
    mean_value = np.mean(data)
    new_dat = []
    for dat in data:
        if abs(dat - np.mean(data)) > m * np.std(data):
            dat = mean_value
        new_dat.append(dat)
    return new_dat

# check if still outliers
data = df['Speed']
data.isnull().sum()
df['Speed'] = replace_outliers(data, 3)
df.sort_index(inplace=True)

# convert time
dt = df.index.values.astype('datetime64[s]')
df = pd.DataFrame(df.values, index = dt)
df.columns = ['Speed']

df.head(5)
df.max()

df.plot()
# plt.show()
plt.title("Time-series plot for segment" + str(chosen_segments[segment_id]))
plt.xlabel('Time')
plt.ylabel('Average Speed')
plt.savefig('time-series-seg'+str(segment_id)+'.png')


# remove seasonality by subtracting all values by the first season [lenghth = 144]
len(df)/float(144) # can be divided by 144
df_d = df.diff(144)
df_d

# after that, plot again.
df_d.plot()
# plt.show()
plt.title("Differentiated Time-series plot for segment" + str(chosen_segments[segment_id]))
plt.xlabel('Time')
plt.ylabel('Average Speed')
plt.savefig('diff-time-series-seg'+str(segment_id)+'.png')

df_d.dropna(inplace=True) # remove all nan
df = df_d

def unit_root_test(ts):
    result = adfuller(ts)
    print('ADF statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values: ')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

unit_root_test(df['Speed'])

# Conclusion: stationary

# ===============================================================================
# 							Plot the ACF
# ===============================================================================

segment_id = 0
def multiple_correlation_plot(ts, lag_list, type):
    i = 0
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    if type == 'acf':
        fig.suptitle('ACF')
        for lag in lag_list:
            sm.graphics.tsa.plot_acf(ts, lags= lag, ax=axes[i])
            axes[i].set_title("")
            i += 1
        plt.savefig('ACF_' + 'SEG_' + str(segment_id) + 'lag_' + str(lag_list[0]) + '.png')
    elif type =='pacf':
        fig.suptitle('PACF')
        for lag in lag_list:
            sm.graphics.tsa.plot_pacf(ts, lags= lag, ax=axes[i])
            axes[i].set_title("")
            i += 1
        plt.savefig('PACF_' + 'SEG_' + str(segment_id) + 'lag_' + str(lag_list[0]) + '.png')
    return

multiple_correlation_plot(df, [10, 20, 30], 'acf')
multiple_correlation_plot(df, [50, 60, 70], 'acf')
multiple_correlation_plot(df, [10, 20, 30], 'pacf')
multiple_correlation_plot(df, [50, 60, 70], 'pacf')

multiple_correlation_plot(df, [100, 200, 300], 'acf')
multiple_correlation_plot(df, [100, 200, 300], 'pacf')

# ===============================================================================
# 							Estimation
# ===============================================================================

# split data
cut_off = int(len(df.index) * 2/3)
train_df = df.iloc[0:cut_off]
test_df = df.iloc[cut_off:]


# fit model
def fit_arma(ts, p, q):
    model = ARMA(ts, order=(p,q))
    model_fit = model.fit(disp=0)
    return model_fit.summary()

with open('{}-results.txt'.format(segment_id), 'a') as f:
    for p in range(1, 5, 1):
        f.write(str(fit_arma(train_df['Speed'], p, 0)))
        f.write('\n')
    for q in range(1, 4, 1):
        f.write(str(fit_arma(train_df['Speed'], 0, q)))
        f.write('\n')
    for p in range(1, 5, 1):
        for q in range(1, 4, 1):
            f.write(str(fit_arma(train_df['Speed'], p, q)))
            f.write('\n')

from matplotlib import pyplot
# select model to be ARMA(1, 6)
model = ARMA(train_df['Speed'], order=(4,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# out-of-sample prediction
start_index = train_df.index[-1]
end_index = test_df.index[-1]
forecast = model_fit.predict(start=start_index, end=end_index)
forecast = forecast[test_df.index] # some days are missing.
pred_error = test_df['Speed'] - forecast
pred_error
MSE = np.square(pred_error).sum()
MSE
# ================================================================================
#                   Plot residuals -- 4 plot
# ================================================================================

# Run sequence plot
# Source : https://github.com/softdevteam/libkalibera/blob/master/python/pykalibera/graphs.py
def run_sequence_plot(data, title="Run sequence plot", filename=None,
        xlabel="Run #", ylabel="Time(s)"):
    """Plots a run sequence graph.
    Arguments:
    data -- list of data points
    Keyword arguments:
    title -- graph title
    filename -- filename to write graph to (None plots to screen)
    xlabel -- label on x-axis"
    ylabel -- label on y-axis"
    """
    xs = range(len(data))
    plt.cla()
    p = plt.plot(xs, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

# Lag plot
# Source: https://github.com/softdevteam/libkalibera/blob/master/python/pykalibera/graphs.py
def my_lag_plot(data, filename=None,
        title=None, xlabel="Lag time(s)", ylabel="Time(s)"):
    """Generates a lag plot.
    Arguments:
    data -- list of data points
    Keyword arguments:
    lag -- which lag to plot
    filename -- filename to write graph to (None plots to screen)
    title -- graph title (if None, then "Lag %d plot" % lag is used)
    xlabel -- label on x-axis
    ylabel -- label on y-axis
    """
    if title is None:
        title = "Lag plot"
    plt.cla()
    p = lag_plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

# Histogram
def hist_plot(data, filename=None, title=None):
    """Generates a histogram.
    Arguments:
    data -- list of data points
    Keyword arguments:
    filename -- filename to write graph to (None plots to screen)
    title -- graph title (if None, then "Histogram" is used)
    xlabel -- label on x-axis
    ylabel -- label on y-axis
    """
    if title is None:
        title = "Histogram"
    plt.cla()
    p = data.hist()
    plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


# Normal probability plot
def qq_plot(data, filename=None, title=None):
    """Generates a histogram.
    Arguments:
    data -- list of data points
    Keyword arguments:
    filename -- filename to write graph to (None plots to screen)
    title -- graph title (if None, then "Normal QQ plot" is used)
    xlabel -- label on x-axis
    ylabel -- label on y-axis
    """
    if title is None:
        title = "Normal QQ plot"
    plt.cla()
    p = sm.qqplot(data, line='45')
    plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


# Actual plotting
run_sequence_plot(pred_error, title="Time-Series-Run sequence plot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ts-rsp.png")

my_lag_plot(pred_error, title="Time-series-lag-plot-SEG" + str(segment_id)+ "-lag" + str(segment_id), filename="SEG"+str(segment_id)+"-ts-lag"+str(segment_id)+".png")

hist_plot(pred_error, title="Time-Series-histogram-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ts-histogram.png")
qq_plot(pred_error, title="Time-Series-qqplot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ts-qqplot.png")


# ================================================================================
#                        Goodness of fit
# ================================================================================

def chisq_goodness_of_fit(true, pred, num_bins):
    min_val = np.array([true.min(), pred.min()]).min()
    max_val = np.array([true.max(), pred.max()]).max()
    true_binned = np.histogram(true, num_bins, (min_val, max_val))[0]
    pred_binned = np.histogram(pred, num_bins, (min_val, max_val))[0]
    # remove zeros
    non_zero_index = []
    for i in range(len(true_binned)):
        if true_binned[i] != 0 and pred_binned[i] != 0:
            non_zero_index.append(i)
    true_binned = true_binned[non_zero_index]
    pred_binned = pred_binned[non_zero_index]
    return chisquare(true_binned, pred_binned)

chisq_goodness_of_fit(test_df['Speed'], forecast, 200)



