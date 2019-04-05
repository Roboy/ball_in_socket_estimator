import numpy as np
from keras.models import model_from_json
import tensorflow
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg


model_name = 'model'

rospy.init_node('3dof predictor', anonymous=True)

# load json and create model
json_file = open('/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/'+model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/"+model_name+".h5") #_checkpoint
rospy.loginfo("Loaded model from disk")
model.summary()

class ball_in_socket_estimator:
    graph = tensorflow.get_default_graph()
    joint_state = rospy.Publisher('/joint_states', sensor_msgs.msg.JointState , queue_size=1)
    def __init__(self):
        # load json and create model
        json_file = open('/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/model.h5")
        print("Loaded model from disk")
        self.listener()
    def magneticsCallback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.x[1], data.y[1], data.x[2], data.y[2]])
        x_test=x_test.reshape((1,6))
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            euler = self.model.predict(x_test)
            #            pos = self.model.predict(x_test)
            rospy.loginfo_throttle(1, (euler[0,0],euler[0,1],euler[0,2]))
            msg = sensor_msgs.msg.JointState()
            msg.header = std_msgs.msg.Header()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['sphere_axis0', 'sphere_axis1', 'sphere_axis2']
            msg.position = [euler[0,0], euler[0,1], euler[0,2]]
            msg.velocity = [0,0,0]
            msg.effort = [0,0,0]
            self.joint_state.publish(msg)

    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.magneticsCallback)
        rospy.spin()

estimator = ball_in_socket_estimator()