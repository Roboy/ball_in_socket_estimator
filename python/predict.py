import os.path

import numpy as np
import rospkg
import rospy
import joblib
import std_msgs.msg
import sensor_msgs.msg
from visualization_msgs.msg import Marker
from roboy_middleware_msgs.msg import MagneticSensor
from nn_model import NeuralNetworkModel
from utils import BodyPart, MagneticId


rospy.init_node('ball_socket_neural_network')
sensors_scaler = [None for _ in MagneticId]
model = [None for _ in MagneticId]
prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
trackingPublisher = rospy.Publisher("/roboy/pinky/sensing/external_joint_states", sensor_msgs.msg.JointState)
targetPublisher = rospy.Publisher("/roboy/pinky/control/joint_targets", sensor_msgs.msg.JointState)


def load_data(name, id):
    global sensors_scaler
    sensors_scaler[id] = joblib.load(name+'/scaler.pkl')


def sin_cos_to_angle(sin_cos):
    return np.array([
        np.arctan2(sin_cos[:, 0], sin_cos[:, 3]),
        np.arctan2(sin_cos[:, 1], sin_cos[:, 4]),
        np.arctan2(sin_cos[:, 2], sin_cos[:, 5]),
    ]).T


def magentic_data_callback(data):

    if sensors_scaler[data.id] is None:
        return

    x_test = np.array(
        [data.x[0], data.y[0], data.z[0],
         data.x[1], data.y[1], data.z[1],
         data.x[2], data.y[2], data.z[2],
         data.x[3], data.y[3], data.z[3]])

    rospy.loginfo_throttle(1, x_test)

    x_test = x_test.reshape((1, len(x_test)))
    x_test = sensors_scaler[data.id].transform(x_test).astype('float32')

    output = model[data.id].predict(x_test)
    output = sin_cos_to_angle(output)[0]

    msg = sensor_msgs.msg.JointState()
    msg.header = std_msgs.msg.Header()
    msg.header.stamp = rospy.Time.now()

    msg.name = [BodyPart[MagneticId(data.id).name] + "_axis" + str(i) for i in range(3)]
    msg.position = [output[0], output[1], output[2]]
    msg.velocity = [0, 0, 0]
    msg.effort = [0, 0, 0]

    trackingPublisher.publish(msg)


if __name__ == '__main__':
    rate = rospy.Rate(100)

    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator') + '/python/'

    for i in range(len(model)):
        body_part = BodyPart[MagneticId(i).name]
        model_path = './output/'+body_part+'_tanh'
        if os.path.isdir(model_path):
            model[i] = NeuralNetworkModel(name=body_part)
            print("Loading model " + model_path + " from disk")
            model[i].restore_model(model_path + "/best_model")
            print("Loaded model " + model_path + " from disk")
            load_data(model_path, i)

    rospy.Subscriber("/roboy/pinky/middleware/MagneticSensor", MagneticSensor, magentic_data_callback, queue_size=1)
    rospy.spin()

