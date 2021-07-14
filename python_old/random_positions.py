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

rospy.init_node('head_random_pos')


class MoveAround(State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done'])
        self.axis0 = rospy.Publisher('/sphere_head_axis0/sphere_head_axis0/target', std_msgs.msg.Float32 , queue_size=1)
        self.axis1 = rospy.Publisher('/sphere_head_axis1/sphere_head_axis1/target', std_msgs.msg.Float32 , queue_size=1)
        self.axis2 = rospy.Publisher('/sphere_head_axis2/sphere_head_axis2/target', std_msgs.msg.Float32 , queue_size=1)
    def execute(self, userdata):
        rospy.loginfo('new pose')
        self.axis0.publish(random.uniform(-0.1, 0.1))
        self.axis1.publish(random.uniform(-0.05, 0.05))
        self.axis2.publish(random.uniform(-0.5, 0.2))
        time.sleep(random.uniform(20, 50))
        return 'done'
    axis0 = []
    axis1 = []
    axis2 = []

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
