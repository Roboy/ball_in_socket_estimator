#!/usr/bin/env python
import roslib
import rospy
import std_msgs
import time

rospy.init_node('head_random_pos')

sphere0 = rospy.Publisher('/sphere_head_axis0/sphere_head_axis0/target', std_msgs.msg.Float32 , queue_size=1)
sphere1 = rospy.Publisher('/sphere_head_axis1/sphere_head_axis1/target', std_msgs.msg.Float32 , queue_size=1)
sphere2 = rospy.Publisher('/sphere_head_axis2/sphere_head_axis2/target', std_msgs.msg.Float32 , queue_size=1)

axis

while not rospy.is_shutdown():
