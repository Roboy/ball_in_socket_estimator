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

publish_magnetic_data = False
show_magnetic_field = False

model_name = "400_tanh"

def main():

    global publish_magnetic_data
    global show_magnetic_field

    # load json and create model
    json_file = open('/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/"+model_name+"_checkpoint.h5")
    print("Loaded model from disk")
    model.summary()

    rospy.init_node('replay')
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    magneticSensor_pub = rospy.Publisher('roboy/middleware/MagneticSensor', MagneticSensor, queue_size=1)
    visualization_pub = rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker, queue_size=100)

    dataset = pandas.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)
    dataset = dataset.values[1+1000000:,0:]
    quaternion_set = dataset[0:,0:4]
    sensors_set = dataset[0:,4:13]
    sensors_set = wrangle.mean_zero(pandas.DataFrame(sensors_set)).values
    sample = 0
    samples = len(quaternion_set)
    t = 0
    rate = rospy.Rate(100)
    error = 0
    i = 0
    for (q, s) in itertools.izip(quaternion_set, sensors_set):
        if rospy.is_shutdown():
            return
        s_input = s.reshape((1,9))
        quat = model.predict(s_input)
        rospy.loginfo_throttle(1, (quat[0,0],quat[0,1],quat[0,2],quat[0,3]))
        norm = numpy.linalg.norm(quat)
        quat = (quat[0,0]/norm,quat[0,1]/norm,quat[0,2]/norm,quat[0,3]/norm)
        error = error+numpy.linalg.norm(quat-q)
        broadcaster.sendTransform((0, 0, 0),
                                  q,
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
                msg.x = [s[0], s[3], s[6]]
                msg.y = [s[1], s[4], s[7]]
                msg.z = [s[2], s[5], s[8]]
                magneticSensor_pub.publish(msg)
                t0 = rospy.Time.now()
            print("%d/%d\t\t%.3f%%" % (t, samples, (t/float(samples))*100.0))
        t = t + 1
        rate.sleep()
    error = error/samples
    print("mean squared error: %f" % (error))
    # Signal handler
    rospy.spin()


if __name__ == '__main__':
    main()