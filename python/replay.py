#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
import numpy as np
import pandas
import rospy
import tensorflow
import tf
import geometry_msgs.msg
from pyquaternion import Quaternion
from keras.layers import Dense, Activation
from keras.models import model_from_json
import pandas
import rospy
import tf
import geometry_msgs.msg
import visualization_msgs.msg
from pyquaternion import Quaternion
from roboy_middleware_msgs.msg import MagneticSensor
from visualization_msgs.msg import Marker
import std_msgs.msg
import sys, select
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy
import itertools
import wrangle
import math

publish_magnetic_data = False
show_magnetic_field = False

model_name = "model"

def euler_to_quaternion(roll, pitch, yaw):
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)


    return [qw, qx, qy, qz]

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        z = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        x = math.atan2(R[1,0], R[0,0])
    else :
        z = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        x = 0

    return np.array([x, y, z])

def main():

    global publish_magnetic_data
    global show_magnetic_field

    # load json and create model
    json_file = open('/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/"+model_name+".h5") #_checkpoint
    print("Loaded model from disk")
    model.summary()

    rospy.init_node('replay')
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    magneticSensor_pub = rospy.Publisher('roboy/middleware/MagneticSensor', MagneticSensor, queue_size=1)
    visualization_pub = rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=100)
    rospy.loginfo('loading data')
    dataset = pandas.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)
    dataset = dataset.values[:,0:]
    dataset = dataset[0:1000000,:]
    print(str(len(dataset))+' samples')
    euler_set = np.array(dataset[:,13:16])
    # mean_euler = euler_set.mean(axis=0)
    # std_euler = euler_set.std(axis=0)
    # euler_set = (euler_set - mean_euler) / std_euler
    print('max euler ' + str(np.amax(euler_set)))
    print('min euler ' + str(np.amin(euler_set)))
    sensors_set = np.array([dataset[:,4],dataset[:,5],dataset[:,7],dataset[:,8],dataset[:,10],dataset[:,11]])
    # sensors_set = np.array([dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10],dataset[:,11],dataset[:,12]])
    sensors_set = np.transpose(sensors_set)
    # mean_sensor = sensors_set.mean(axis=0)
    # std_sensor = sensors_set.std(axis=0)
    # sensors_set = (sensors_set - mean_sensor) / std_sensor
    quaternion_set = dataset[0:,0:4]
    # sensors_set = wrangle.mean_zero(pandas.DataFrame(sensors_set)).values
    sample = 0
    samples = len(euler_set)
    t = 0
    rate = rospy.Rate(100)
    error = 0
    stride = 1
    print('model predicts')
    euler_predict = model.predict(sensors_set)
    mse = numpy.linalg.norm(euler_predict-euler_set)/len(euler_predict)
    print('mse: ' + str(mse))
    for i in range(0,len(euler_predict),stride):
        if rospy.is_shutdown():
            return
        # rospy.loginfo_throttle(1, (euler_predict[i,0],euler_predict[i,1],euler_predict[i,2]))
        # error = error+numpy.linalg.norm(euler_predict[i,i:i+stride]-euler_set[i,i:i+stride])
        # euler_p = euler_predict[i,:]*std_euler+mean_euler
        quat = euler_to_quaternion(-euler_predict[i,0],-euler_predict[i,1],euler_predict[i,2])
        euler_predictED = [euler_predict[i,0],euler_predict[i,1],-euler_predict[i,2]]
        euler_truth = rotationMatrixToEulerAngles(Quaternion(quaternion_set[i,:]).rotation_matrix)
        broadcaster.sendTransform((0, 0, 0),
                                  euler_to_quaternion(euler_predictED[0],euler_predictED[1],euler_predictED[2]),#euler_p[0],euler_p[1],euler_p[2]),
                                  rospy.Time.now(),
                                  "predict",
                                  "world")
        broadcaster.sendTransform((0, 0, 0),
                                  euler_to_quaternion(euler_truth[0],euler_truth[1],euler_truth[2]),
                                  rospy.Time.now(),
                                  "ground_truth",
                                  "world")
        # if(t==0):
            # print(quat)
            # print(q)

        if show_magnetic_field:

            msg2 = visualization_msgs.msg.Marker()
            msg2.type = msg2.SPHERE
            msg2.id = i
            msg2.color.r = 1
            msg2.color.a = 1
            msg2.action = msg2.ADD
            msg2.header.seq = i
            msg2.header.frame_id = "world"
            msg2.header.stamp = rospy.Time.now()
            msg2.lifetime = rospy.Duration(0)
            msg2.scale.x = 0.01
            msg2.scale.y = 0.01
            msg2.scale.z = 0.01
            msg2.pose.orientation.w = 1
            msg2.ns = "magnetic_field_sensor0"

            sens = numpy.array([s[3],s[4],s[5]])
            q_w = Quaternion(q)
            sens_w = q_w.rotate(sens) * 0.01
            msg2.pose.position.x = sens_w[0]
            msg2.pose.position.y = sens_w[1]
            msg2.pose.position.z = sens_w[2]
            i= i+1

            visualization_pub.publish(msg2)
        if publish_magnetic_data:
            msg = MagneticSensor()
            msg.id = 5
            msg.sensor_id = [0, 1, 2]
            msg.x = [sensors_set[i,0],sensors_set[i,3]]#,sensors_set[i,6]
            msg.y = [sensors_set[i,1],sensors_set[i,4]]#,sensors_set[i,7]]
            msg.z = [sensors_set[i,2],sensors_set[i,5]]#,sensors_set[i,8]]
            magneticSensor_pub.publish(msg)
            t0 = rospy.Time.now()
        print("%d/%d\t\t%.3f%%" % (t, samples, (t/float(samples))*100.0))
        t = t + stride
        rate.sleep()
    error = error/samples
    print("mean squared error: %f" % (error))
    # Signal handler
    rospy.spin()


if __name__ == '__main__':
    main()
