import rospy
import sensor_msgs.msg, std_msgs
rospy.init_node('joint_targets')
joint_targets_pub = rospy.Publisher('joint_targets', sensor_msgs.msg.JointState, queue_size=100)

pos = [[0,0,0],\
[-0.3,0,0],[-0.15,0,0],[0.15,0,0],[0.3,0,0],\
[0,-0.3,0],[0,-0.15,0],[0,0.15,0],[0,0.3,0],\
[0,0,-0.3],[0,0,-0.15],[0,0,0.15],[0,0,0.3],\
]

rate = rospy.Rate(1)

if not rospy.is_shutdown():
    for p in pos:
        msg = sensor_msgs.msg.JointState()
        msg.position = p
        joint_targets_pub.publish(msg)
        rospy.loginfo('%f %f %f'%(p[0],p[1],p[2]))
        rate.sleep()
