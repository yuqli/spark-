#!usr/bin/python

"""
Precondition: Already load the pickle file and output to CSV for each segment
Then load each segmant separately
"""
# ===============================================================================
#                               Preprocessing
# ===============================================================================
from pyspark.mllib.stat import Statistics

segment_id = 0
# filepath = '/home/yl5090/SEG' + str(i) + '.csv'
filepath = './SEG{}.csv'.format(str(segment_id))
df = read_csv(filepath, names=['Speed'],header=None, index_col=0)
df.fillna(method='ffill', inplace = True)

# convert time
dt = df.index.values.astype('datetime64[s]')
df = pd.DataFrame(df.values, index = dt)
df.columns = ['Speed']

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

data = df['Speed']
data.isnull().sum()
df['Speed'] = replace_outliers(data, 3)
df.sort_index(inplace=True)

df.head(5)
df.max()
plt.hist(df.values, bins = 10)
plt.show()

df_d = df.diff(144)
df_d.dropna(inplace=True) # remove all nan
df = df_d
df

# de-trending data
# Since fit in an ARMA(3, 0) model, now use p = 3
df['t1'] = df.shift(1)
df['t2'] = df['t1'].shift(1)
df['t3'] = df['t2'].shift(1)
df['t4'] = df['t3'].shift(1)
df.drop(df.head(4).index, inplace=True) # remove the first three row

# split data
cut_off = int(len(df.index) * 2/3)
train_df = df.iloc[0:cut_off]
test_df = df.iloc[cut_off:]

sqlCtx = SQLContext(sc)

train_df = sqlCtx.createDataFrame(train_df)
test_df = sqlCtx.createDataFrame(test_df)

# ================================================================================
#                              Transform data into LabelPoint
# ================================================================================
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
temp = train_df.map(lambda line:LabeledPoint(line[0],[line[1], line[2], line[3], line[4]]))
temp_test = test_df.map(lambda line:LabeledPoint(line[0],[line[1], line[2], line[3], line[4]]))

# ================================================================================
#                              OLS regression
# ================================================================================
import statsmodels.api as sm
# fit an ols
lr = LinearRegressionWithSGD.train(temp, 1000,.2, intercept=True)

# evaluate the model on training data
valuesAndPreds = temp.map(lambda p: (float(p.label), float(lr.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))

# Validate the model on test data
valuesAndPreds = temp_test.map(lambda p: (float(p.label), float(lr.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))

pred_error = valuesAndPreds.map(lambda (v, p): (v-p, )).toDF().toPandas()
valuesAndPreds_df = valuesAndPreds.toDF().toPandas()

# Actual plotting
run_sequence_plot(pred_error, title="OLS-Run sequence plot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ols-rsp.png")

from pandas.tools.plotting import lag_plot
my_lag_plot(pred_error,title="OLS-lag-plot-SEG" + str(segment_id)+ "-lag" + str(i), filename="SEG"+str(segment_id)+"-ols-lag"+str(segment_id)+".png")
hist_plot(pred_error, title="OLS-histogram-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ols-histogram.png")
qq_plot(pred_error, title="OLS-qqplot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-ols-qqplot.png")

# now run chi-square
def bin_data(true, pred, num_bins):
    min_val = np.array([true.min(), pred.min()]).min()
    max_val = np.array([true.max(), pred.max()]).max()
    true_binned = np.histogram(true, num_bins, (min_val, max_val))[0]
    pred_binned = np.histogram(pred, num_bins, (min_val, max_val))[0]
    return true_binned, pred_binned

from pyspark.mllib.linalg import Vectors

v, p = bin_data(valuesAndPreds_df['_1'].values, valuesAndPreds_df['_2'], 200)
v = Vectors.dense(v)
p = Vectors.dense(p)
pearson = Statistics.chiSqTest(v, p)
pearson.pValue
pearson.statistic

# ================================================================================
#                              Ridge regression
# ================================================================================

# fit an ridge
lr_ridge = LinearRegressionWithSGD.train(temp, 1000,.2, intercept=True, regType = 'l2')

# evaluate the model on training data
valuesAndPreds = temp.map(lambda p: (float(p.label), float(lr_ridge.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))

# Validate the model on test data
valuesAndPreds = temp_test.map(lambda p: (float(p.label), float(lr_ridge.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))
pred_error = valuesAndPreds.map(lambda (v, p): (v-p, )).toDF().toPandas()
valuesAndPreds_df = valuesAndPreds.toDF().toPandas()

# Actual plotting
run_sequence_plot(pred_error, title="Ridge-Run sequence plot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-Ridge-rsp.png")

from pandas.tools.plotting import lag_plot
my_lag_plot(pred_error,title="Ridge-lag-plot-SEG" + str(segment_id)+ "-lag" + str(i), filename="SEG"+str(segment_id)+"-Ridge-lag.png")

hist_plot(pred_error, title="Ridge-histogram-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-Ridge-histogram.png")
qq_plot(pred_error, title="Ridge-qqplot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-Ridge-qqplot.png")

v, p = bin_data(valuesAndPreds_df['_1'].values, valuesAndPreds_df['_2'], 200)
v = Vectors.dense(v)
p = Vectors.dense(p)
pearson = Statistics.chiSqTest(v, p)
pearson.pValue
pearson.statistic


# ================================================================================
#                              Lasso regression
# ================================================================================

# fit a lasso
lr_lasso = LinearRegressionWithSGD.train(temp, 1000,.2, intercept=True, regType = 'l1')

# evaluate the model on training data
valuesAndPreds = temp.map(lambda p: (float(p.label), float(lr_lasso.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))

# Validate the model on test data
valuesAndPreds = temp_test.map(lambda p: (float(p.label), float(lr_lasso.predict(p.features))))
MSE = valuesAndPreds.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x+y)/valuesAndPreds.count()
print('Mean Squared Error = ' +  str(MSE))
pred_error = valuesAndPreds.map(lambda (v, p): (v-p, )).toDF().toPandas()
valuesAndPreds_df = valuesAndPreds.toDF().toPandas()

# Actual plotting
run_sequence_plot(pred_error, title="lasso-Run sequence plot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-lasso-rsp.png")

from pandas.tools.plotting import lag_plot
my_lag_plot(pred_error,title="lasso-lag-plot-SEG" + str(segment_id)+ "-lag" + str(i), filename="SEG"+str(segment_id)+"-lasso-lag.png")

hist_plot(pred_error, title="lasso-histogram-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-lasso-histogram.png")
qq_plot(pred_error, title="lasso-qqplot-SEG" + str(segment_id), filename="SEG"+str(segment_id)+"-lasso-qqplot.png")

v, p = bin_data(valuesAndPreds_df['_1'].values, valuesAndPreds_df['_2'], 200)
v = Vectors.dense(v)
p = Vectors.dense(p)
pearson = Statistics.chiSqTest(v, p)
pearson.pValue
pearson.statistic

