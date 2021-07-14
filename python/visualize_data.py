import os
import rospy
import sensor_msgs.msg
from utils import BodyPart
import argparse
from argparse import RawTextHelpFormatter

rospy.init_node('visualize_data')

nnPublisher = rospy.Publisher("/roboy/pinky/validate_nn", sensor_msgs.msg.JointState)
gtPublisher = rospy.Publisher("/roboy/pinky/validate_gt", sensor_msgs.msg.JointState)
body_part = None


def extjs_test_callback(data):
    if data.name[0] == body_part + "_axis0":
        nnPublisher.publish(data)


def extjs_callback(data):
    if data.name[0] == body_part + "_axis0":
        gtPublisher.publish(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rqt plot data, use visualize_data.py -h for more information',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--body_part', type=BodyPart, choices=list(BodyPart),
                        help='Body part: head, shoulder_left, shoulder_right, hand_left, hand_right', required=True)
    args = parser.parse_args()

    body_part = args.body_part

    extjs_test_sub = rospy.Subscriber('/roboy/pinky/sensing/external_joint_states', sensor_msgs.msg.JointState,
                                      extjs_test_callback)
    extjs_sub = rospy.Subscriber('/roboy/pinky/external_joint_states', sensor_msgs.msg.JointState, extjs_callback)

    os.system("rqt_plot"
              " /roboy/pinky/validate_nn/position[0]" 
              " /roboy/pinky/validate_nn/position[1]" 
              " /roboy/pinky/validate_nn/position[2]" 
              " /roboy/pinky/validate_gt/position[0]" 
              " /roboy/pinky/validate_gt/position[1]" 
              " /roboy/pinky/validate_gt/position[2]")

    rospy.spin()
