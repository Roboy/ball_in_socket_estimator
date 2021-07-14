import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import wrangle as wr
from pandas import DataFrame
from pandas import concat

from sklearn.model_selection import train_test_split, KFold
# Function to create model, required for KerasClassifier
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataset = pd.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)

dataset = dataset.values[:,0:]
quaternion_set = np.array(dataset[:,0:4])
sensors_set = np.array(dataset[:,4:13])
sensors_set = wr.mean_zero(pd.DataFrame(sensors_set)).values
data_in_train = sensors_set[:int(len(sensors_set)*0.7),:]
data_in_test = sensors_set[int(len(sensors_set)*0.7):,:]
data_out_train = quaternion_set[:int(len(sensors_set)*0.7),:]
data_out_test = quaternion_set[int(len(sensors_set)*0.7):,:]
# frame as supervised learning
look_back_samples = 1
train_X = series_to_supervised(data_in_train, look_back_samples, 0).values
test_X = series_to_supervised(data_in_test, look_back_samples, 0).values
train_y = series_to_supervised(data_out_train, 1, 0).values[look_back_samples-1:,:]
test_y = series_to_supervised(data_out_test, 1, 0).values[look_back_samples-1:,:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def create_model(activation='relu',neurons=100, dropout=0):
    # model = Sequential()
    # model.add(Dense(units=neurons, input_dim=9,kernel_initializer='normal', activation=activation))
    # model.add(Dropout(dropout))
    # model.add(Dense(units=neurons, kernel_initializer='normal', activation=activation))
    # model.add(Dropout(dropout))
    # model.add(Dense(units=4,kernel_initializer='normal'))
    # model.compile(loss='mean_squared_error',
    #               optimizer='adam',
    #               metrics=['acc'])

    model = Sequential()
    model.add(LSTM(units=neurons, input_shape=(train_X.shape[1], train_X.shape[2]),activation="relu"))
    model.add(Dense(units=neurons, kernel_initializer='normal', activation="relu"))
    model.add(Dense(train_y.shape[1], activation="relu"))
    model.compile(loss='mse', optimizer='adam')
    return model


# create model
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=400, verbose=0, shuffle=False)
# define the grid search parameters
batch_size = [500]
epochs = [50]
neurons = [100,200,300,400]
activation = ['tanh']
dropout = [0,0.1,0.2,0.3]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=25, verbose=50, scoring='neg_mean_squared_error')
grid_result = grid.fit(train_X, train_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# print("TestScore: %f" % (grid_result.score(x,y)))

# serialize model to JSON
model_json = grid_result.best_estimator_.model.to_json()
with open("beschde.json", "w") as json_file:
    json_file.write(model_json)
grid_result.best_estimator_.model.save('beschde.h5')





