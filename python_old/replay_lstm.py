#!/usr/bin/env python
from keras.models import Sequential
import numpy as np
import pandas
from pandas import DataFrame
from pandas import concat
import rospy
import tensorflow
import tf
import geometry_msgs.msg
from pyquaternion import Quaternion
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import model_from_json
from roboy_middleware_msgs.msg import MagneticSensor
from visualization_msgs.msg import Marker
import sys, select
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy
import wrangle as wr
import itertools
from matplotlib import pyplot
global record
global sensors_set
global quaternion_set

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

def main():

    global publish_magnetic_data
    global show_magnetic_field

    # load json and create model
    json_file = open('/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/beschde.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/beschde.h5")
    print("Loaded model from disk")

    rospy.init_node('replay')
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    dataset = pandas.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)

    dataset = dataset.values[0:500000,0:]
    quaternion_set = np.array(dataset[:,0:4])
    sensors_set = np.array(dataset[:,4:13])
    sensors_set = wr.mean_zero(pandas.DataFrame(sensors_set)).values
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
    sample = 0
    samples = len(quaternion_set)
    t = 0
    rate = rospy.Rate(100)
    error = 0
    i = 0
    for (data_in, data_out) in itertools.izip(train_X, train_y):
        if rospy.is_shutdown():
            return
        data_in = data_in.reshape((1,data_in.shape[0],data_in.shape[1]))
        quat = model.predict(data_in)
        rospy.loginfo_throttle(1, (quat[0,0],quat[0,1],quat[0,2],quat[0,3]))
        norm = numpy.linalg.norm(quat)
        quat = (quat[0,0]/norm,quat[0,1]/norm,quat[0,2]/norm,quat[0,3]/norm)
        error = error+numpy.linalg.norm(quat-data_out)
        broadcaster.sendTransform((0, 0, 0),
                                  data_out,
                                  rospy.Time.now(),
                                  "ground_truth",
                                  "world")
        broadcaster.sendTransform((0, 0, 0),
                                  quat,
                                  rospy.Time.now(),
                                  "predict",
                                  "world")
        if(t%100==0):
            # print(quat)
            # print(q)

            print("%d/%d\t\t%.3f%%" % (t, samples, (t/float(samples))*100.0))
        t = t + 1
        rate.sleep()
    error = error/samples
    print("mean squared error: %f" % (error))
    # Signal handler
    rospy.spin()


if __name__ == '__main__':
    main()
