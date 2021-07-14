import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import calendar
from datetime import datetime

from keras import callbacks
import os, itertools

from sklearn.model_selection import train_test_split, KFold
# Function to create model, required for KerasClassifier
def create_model(activation='relu',neurons=100, dropout=0):
    model = Sequential()
    model.add(Dense(units=neurons, input_dim=12,kernel_initializer='normal', activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(units=neurons, kernel_initializer='normal', activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(units=3,kernel_initializer='normal'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


dt = datetime.utcnow()
timestamp = calendar.timegm(dt.utctimetuple())


class KerasRegressorTB(KerasRegressor):

    def __init__(self, *args, **kwargs):
        super(KerasRegressor, self).__init__(*args, **kwargs)

    def fit(self, x, y, log_dir='./hyperparameter_logs/'+str(timestamp), **kwargs):
        cbs = None
        if log_dir is not None:
            # Make sure the base log directory exists
            try:
                os.makedirs(log_dir)
            except OSError:
                pass
            params = self.get_params()
            conf = ",".join("{}={}".format(k, params[k])
                            for k in sorted(params))
            conf_dir_base = os.path.join(log_dir, conf)
            # Find a new directory to place the logs
            for i in itertools.count():
                try:
                    conf_dir = "{}_split-{}".format(conf_dir_base, i)
                    os.makedirs(conf_dir)
                    break
                except OSError:
                    pass
            cbs = [callbacks.TensorBoard(log_dir=conf_dir, histogram_freq=0,
                               write_graph=True, write_images=False),
                   callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min')]
        super(KerasRegressor, self).fit(x, y, callbacks=cbs)#, **kwargs

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = pd.read_csv(os.environ['HOME']+"/workspace/roboy3/head_data0.log", delim_whitespace=True, header=1)
dataset = dataset.values[1:len(dataset)-1,0:]
print('%d values'%(len(dataset)))
# dataset = dataset[abs(dataset[:,12])<=0.7,:]
# dataset = dataset[abs(dataset[:,13])<=0.7,:]
# dataset = dataset[abs(dataset[:,14])<=1.5,:]
# dataset = dataset[abs(dataset[:,12])!=0.0,:]
# dataset = dataset[abs(dataset[:,13])!=0.0,:]
# dataset = dataset[abs(dataset[:,14])!=0.0,:]
# print('%d values after filtering outliers'%(len(dataset)))
# data_split = 1
# train_set = dataset[:int(len(dataset)*data_split),:]
# numpy.random.shuffle(train_set)
euler_set = numpy.array(dataset[:,12:15])
sensors_set = numpy.array([dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10],dataset[:,11]])
sensors_set = numpy.transpose(sensors_set)
y = euler_set
x = sensors_set
print(x[0])
print(y[0])
# x = wr.mean_zero(pd.DataFrame(x)).values
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# create model
cbs = [callbacks.TensorBoard(log_dir='./log_hyperparameter', histogram_freq=0,
                             write_graph=True, write_images=False),
       callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min')]
model = KerasRegressorTB(build_fn=create_model ,verbose=0, epochs=1000, validation_split=0.7, shuffle=True)
# define the grid search parameters
# batch_size = [500]
# epochs = [60]
neurons = [50,100,150,200]
activation = ['relu']
dropout = [0,0.05,0.1]
batch_size = [200,400,600,800]
param_grid = dict(neurons=neurons,activation=activation, batch_size=batch_size, dropout=dropout)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=60, verbose=50, scoring='neg_mean_squared_error',fit_params={'callbacks': cbs})#,fit_params={'log_dir': './log_hyperparameter'}
grid_result = grid.fit(x, y)
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

# import talos as ta
# import wrangle as wr
# from talos.metrics.keras_metrics import fmeasure_acc
# from talos import live
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.activations import relu, elu, sigmoid
# import pandas as pd
# from talos.utils.gpu_utils import parallel_gpu_jobs
# from talos.utils.gpu_utils import force_cpu
#
# def msj_shoulder_model(x_train, y_train, x_val, y_val, params):
#     model = Sequential()
#     model.add(Dense(params['first_layer_number_neurons'], input_dim=x_train.shape[1], kernel_initializer=params['kernel_initializer'], activation=params['activation']))
#     model.add(Dense(params['second_layer_number_neurons'], kernel_initializer=params['kernel_initializer'], activation=params['activation']))
#     model.add(Dense(params['third_layer_number_neurons'], kernel_initializer=params['kernel_initializer'], activation=params['activation']))
#     model.add(Dropout(params['dropout']))
#     model.add(Dense(y_train.shape[1], kernel_initializer=params['kernel_initializer'], activation=params['last_activation']))
#
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam',
#                   metrics=['acc', fmeasure_acc])
#
#     out = model.fit(x_train, y_train,
#                     batch_size=params['batch_size'],
#                     epochs=params['epochs'],
#                     verbose=0,
#                     callbacks=[live()],
#                     validation_data=[x_val, y_val])
#
#     return out, model
#
# dataset = pd.read_csv("/home/roboy/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)
# dataset = dataset.values[1:,0:]
# y_train = dataset[0:1000,0:4]
# x_train = dataset[0:1000,4:13]
# y_valid = dataset[1000:2000,0:4]
# x_valid = dataset[1000:2000,4:13]
# # sensor_set = wr.mean_zero(pd.DataFrame(sensor_set)).values
#
# p = {
#     'first_layer_number_neurons': [10],
#     'second_layer_number_neurons': [10],
#     'third_layer_number_neurons': [10],
#     'activation': [relu, elu, sigmoid],
#     'last_activation': [relu, elu, sigmoid],
#     'batch_size': [500],
#     'kernel_initializer': ['uniform','normal'],
#     'dropout': [0],
#     'epochs': [30]
# }
#
# # Force CPU use on a GPU system
# force_cpu()
#
#
# h = ta.Scan(x=x_train,
#             y=y_train,
#             model=msj_shoulder_model,
#             params=p,
#             dataset_name='left_shoulder',
#             experiment_no='1',
#             print_params=True)
#
# e = ta.Evaluate(h)
# e.evaluate(x_valid, y_valid, folds=10, average='samples')
