#!/usr/bin/env python
import roslib

roslib.load_manifest('smach')
import rospy

import threading
import random
import smach
from smach import StateMachine, Concurrence, State, Sequence
from smach_ros import ServiceState, SimpleActionState, IntrospectionServer
import std_msgs
import time
import math

rospy.init_node('msj_platform_random_pos')


class MoveAround(State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'])
        self.axis0 = rospy.Publisher('/sphere_axis0/sphere_axis0/target', std_msgs.msg.Float32 , queue_size=100)
        self.axis1 = rospy.Publisher('/sphere_axis1/sphere_axis1/target', std_msgs.msg.Float32 , queue_size=100)
        self.axis2 = rospy.Publisher('/sphere_axis2/sphere_axis2/target', std_msgs.msg.Float32 , queue_size=100)
    def execute(self, userdata):
        rospy.loginfo('new pose')
        self.axis0.publish(math.sin(self.t/100)*0.2)
        self.axis0.publish(math.cos(self.t/100)*0.2)
        self.t = self.t+1
        # self.axis1.publish(random.uniform(-0.565487, 0.188496))
        # self.axis2.publish(random.uniform(-0.879646, 0.879646))
        time.sleep(0.1)
        return 'done'
    axis0 = []
    axis1 = []
    axis2 = []
    t = 0

def main():
    # Create the top level SMACH state machine
    sm_top = smach.StateMachine(outcomes=['success'])
    sm_top.userdata.cup = 0

    # Open the container
    with sm_top:
        smach.StateMachine.add('MOVEAROUND', MoveAround(),
                               transitions={'done': 'MOVEAROUND'})

    # Execute SMACH plan
    outcome = sm_top.execute()

    # Attach a SMACH introspection server
    sis = IntrospectionServer('msj_move_around', sm_top, '/MOVEAROUND')
    sis.start()

    # Execute SMACH tree in a separate thread so that we can ctrl-c the script
    smach_thread = threading.Thread(target=sm_top.execute)
    smach_thread.start()

    # Signal handler
    rospy.spin()


if __name__ == '__main__':
    main()