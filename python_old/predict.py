import numpy as np
from keras.models import model_from_json
import tensorflow
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg
import rospkg
import math
import sys, select
from visualization_msgs.msg import Marker
import geometry_msgs.msg
from pyquaternion import Quaternion
import magjoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("body_part",help="ball joint prefix, eg head")
parser.add_argument("sensor_id",help="the id of the ball joint sensor",type=int,default=0)
args = parser.parse_args()
print(args)

ball = magjoint.BallJoint(args.config)

rospy.init_node('shoulder_predictor',anonymous=True)
normalize_magnetic_strength = False

if len(sys.argv) < 2:
    print("\nUSAGE: python predict.py config body_part id, e.g. \n python3 predict.py single_magnet.yaml shoulder_left 0\n")
    sys.exit()

body_part = args.body_part
id = args.sensor_id

print("prediction for %s with id %d"%(body_part,id))
if normalize_magnetic_strength:
    rospy.logwarn("normalizing magnetic field")

class ball_in_socket_estimator:
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator')+'/python/'
    roll = 0
    pitch = 0
    yaw = 0
    # base_path= '/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/'
    network_name = body_part+'model'
    model_name =  body_part+'model'
    model_to_publish_name = body_part
    offset = [0,0,0]
    graph = tensorflow.get_default_graph()
    # base_path= '/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/'
    joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState , queue_size=1)
    msg = sensor_msgs.msg.JointState()
    prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    # selection = [0, 4, 8, 12]
    selection = [0,1,2,3]
    def __init__(self):
        # load json and create model
        json_file = open(self.base_path+self.network_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.base_path+self.model_name+".h5")
        print("Loaded model from disk: "+self.base_path+self.model_name+".h5")
        self.model.summary()
        self.msg.position = [0,0,0]
        self.listener()

    def trackingCallback(self,data):
        # if (self.body_part == "shoulder_right" and data.id == 4) or (self.body_part == "shoulder_left" and data.id == 3):
        self.roll = data.position[0]
        self.pitch = data.position[1]
        self.yaw = data.position[2]
        # rospy.loginfo_throttle(5, "receiving tracking data")

    def publishErrorCube(self,error_roll, error_pitch, error_yaw):
        msg2 = Marker()
        msg2.header = std_msgs.msg.Header()
        msg2.header.stamp = rospy.Time.now()
        msg2.action = msg2.ADD
        msg2.ns = 'prediction error'
        msg2.id = 19348720
        msg2.type = msg2.CUBE
        msg2.scale.x = abs(error_roll)/10.0
        msg2.scale.y = abs(error_pitch)/10.0
        msg2.scale.z = abs(error_yaw)/10.0
        msg2.header.frame_id = 'world'
        msg2.color.a = 0.3
        if error_pitch>1 or error_roll>1 or error_yaw >1:
            msg2.color.r = 1
        else:
            msg2.color.g = 1
        msg2.pose.orientation.w = 1
        msg2.pose.position.z = 0.4
        self.prediction_pub.publish(msg2)
    def publishErrorText(self,error_roll, error_pitch, error_yaw):
        string = "%.3f %.3f %.3f" % (error_roll, error_pitch, error_yaw)
        color = std_msgs.msg.ColorRGBA(0,1,0, 1.0)
        if error_pitch>1 or error_roll>1 or error_yaw >1:
            color = std_msgs.msg.ColorRGBA(1.0, 0, 0, 1.0)
        marker = Marker(
        type=Marker.TEXT_VIEW_FACING,
        id=0,
        lifetime=rospy.Duration(1.5),
        pose=geometry_msgs.msg.Pose(geometry_msgs.msg.Point(0,0,0.4), geometry_msgs.msg.Quaternion(0, 0, 0, 1)),
        scale=geometry_msgs.msg.Vector3(0.01, 0.01, 0.01),
        header=std_msgs.msg.Header(frame_id='world'),
        color=color,
        text=string)
        self.prediction_pub.publish(marker)

    def magneticsCallback(self, data):
        if(data.id == id):
            values = []
            for select in self.selection:
                val = np.array((data.x[select], data.y[select], data.z[select]))
                sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-ball.config['sensor_angle'][select][2])
                val = sensor_quat.rotate(val)
                if normalize_magnetic_strength:
                    val /= np.linalg.norm(val)
                values.append(val[0])
                values.append(val[1])
                values.append(val[2])
            x_test = np.array(values)#np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2], data.x[3], data.y[3], data.z[3]])
            x_test=x_test.reshape((1,12))
            rospy.loginfo_throttle(5,"mag data: \n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f"%(\
                values[0],values[1],values[2],\
                values[3],values[4],values[5],\
                values[6],values[7],values[8],\
                values[9],values[10],values[11])\
            )
            with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
                euler = self.model.predict(x_test)
                euler = [euler[0,0],euler[0,1],euler[0,2]]

                self.msg.header = std_msgs.msg.Header()
                self.msg.header.stamp = rospy.Time.now()
                self.msg.name = [self.model_to_publish_name+'_axis0', self.model_to_publish_name+'_axis1', self.model_to_publish_name+'_axis2']
                # if self.model_to_publish_name=="head":
                #     self.msg.position = [-euler[0], euler[2], euler[1]]
                # elif self.model_to_publish_name=="wrist_left":
                #     self.msg.position = [-euler[1], -euler[0], -euler[2]]
                # elif self.model_to_publish_name=="shoulder_left":
                #     self.msg.position = [euler[1], euler[0], euler[2]]#[euler[1]+51.94/180*math.pi, euler[0]+70.96/180*math.pi, euler[2]-35.95/180*math.pi]
                # else:
                self.msg.position = [euler[0], euler[1], euler[2]]

                rospy.loginfo_throttle(1, (self.msg.position[0],self.msg.position[1],self.msg.position[2]))

                for i in range(len(self.msg.position)):
                    self.msg.position[i] += self.offset[i]
                self.msg.velocity = [0,0,0]
                self.msg.effort = [0,0,0]
                self.joint_state.publish(self.msg)

                error_roll = (((self.roll-euler[0])*180.0/math.pi)**2)**0.5
                error_pitch = (((self.pitch-euler[1])*180.0/math.pi)**2)**0.5
                error_yaw = (((self.yaw-euler[2])**2)*180.0/math.pi)**0.5
                rospy.loginfo_throttle(1, "predict: %f %f %f, truth %f %f %f\nerror %f %f %f"%(self.msg.position[0]*180.0/math.pi,self.msg.position[1]*180.0/math.pi,self.msg.position[2]*180.0/math.pi,self.roll*180.0/math.pi,self.pitch*180.0/math.pi,self.yaw*180.0/math.pi,error_roll,error_pitch,error_yaw))

                self.publishErrorCube(error_roll,error_pitch,error_yaw)
                self.publishErrorText(error_roll,error_pitch,error_yaw)

    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.magneticsCallback)
        trackingSubscriber = rospy.Subscriber("joint_targets", sensor_msgs.msg.JointState, self.trackingCallback, queue_size=1)
        rospy.spin()

estimator = ball_in_socket_estimator()
