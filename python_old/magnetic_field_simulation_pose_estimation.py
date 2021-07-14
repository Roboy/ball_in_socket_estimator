import numpy as np
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg
import rospkg
import matplotlib
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random
import MDAnalysis
import MDAnalysis.visualization.streamlines_3D
import mayavi
from mayavi import mlab
import math, os, time, sys
import geometry_msgs
from multiprocessing import Pool, freeze_support
from pathlib import Path
from yaml import load,dump,Loader,Dumper

rospy.init_node('magnetic_field_pose_estimation')
joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState , queue_size=1)
b_target = [(0,0,0),(0,0,0),(0,0,0),(0,0,0)]


normalize_magnetic_strength = False

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_field_simulation_pose_estimation.py balljoint_config_yaml body_part, e.g. \n python3 magnetic_field_simulation_pose_estimation.py test.yaml head\n")
    sys.exit()

balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
body_part = sys.argv[2]

def gen_sensors(pos,pos_offset,angle,angle_offset):
    sensors = []
    i = 0
    for p in pos:
        s = Sensor(pos=(pos[i][0]+pos_offset[i][0],pos[i][1]+pos_offset[i][1],pos[i][2]+pos_offset[i][2]))
        s.rotate(angle=angle[i][0]+angle_offset[i][0],axis=(1,0,0))
        s.rotate(angle=angle[i][1]+angle_offset[i][1],axis=(0,1,0))
        s.rotate(angle=angle[i][2]+angle_offset[i][2],axis=(0,0,1))
        sensors.append(s)
        i = i+1
    return sensors

def gen_magnets(field_strength,mag_dim,pos,pos_offset,angle,angle_offset):
    magnets = []
    i = 0
    for field in field_strength:
        magnet = Box(mag=(0,0,field), \
         dim=mag_dim[i],\
         pos=(pos[i][0]+pos_offset[i][0],pos[i][1]+pos_offset[i][1],pos[i][2]+pos_offset[i][2]))
        magnet.rotate(angle=angle[i][0]+angle_offset[i][0],axis=(1,0,0))
        magnet.rotate(angle=angle[i][1]+angle_offset[i][1],axis=(0,1,0))
        magnet.rotate(angle=angle[i][2]+angle_offset[i][2],axis=(0,0,1))
        magnets.append(magnet)
        i = i+1
    return magnets

sensors = gen_sensors(balljoint_config['sensor_pos'],balljoint_config['sensor_pos_offsets'],balljoint_config['sensor_angle'],balljoint_config['sensor_angle_offsets'])


def func(x):
    global b_target
    magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                    balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
    magnets.rotate(x[0],(1,0,0), anchor=(0,0,0))
    magnets.rotate(x[1],(0,1,0), anchor=(0,0,0))
    magnets.rotate(x[2],(0,0,1), anchor=(0,0,0))
    b_error = 0
    i = 0
    for sens in sensors:
        val = sens.getB(magnets)
        if normalize_magnetic_strength:
            val /= np.linalg.norm(val)
        b_error = b_error + np.linalg.norm(val-b_target[i])#np.dot(sens.getB(magnets),b_target[i])#,
        i=i+1
    # print(b_error)
    return [b_error]

pos_estimate_prev = [0,0,0]

def magneticsCallback(data):
    if(data.id != balljoint_config['id']):
        return
    global pos_estimate_prev
    for i in range(0,4):
        val = np.array((data.x[i], data.y[i], data.z[i]))
        if normalize_magnetic_strength:
            val /= np.linalg.norm(val)
        b_target[i] = val
    # print(b_target)
    res = least_squares(func, pos_estimate_prev, bounds = ((-50,-50,-80), (50, 50, 80)),ftol=1e-3, xtol=1e-3)#,max_nfev=20
    b_field_error = res.cost
    rospy.loginfo_throttle(1,"result %.3f %.3f %.3f b-field error %.3f"%(res.x[0],res.x[1],res.x[2],res.cost))
    # print("result %.3f %.3f %.3f b-field error %.3f\nb_target %.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f"%(res.x[0],res.x[1],res.x[2],res.cost,b_target[0][0],b_target[0][1],b_target[0][2],b_target[1][0],b_target[1][1],b_target[1][2],b_target[2][0],b_target[2][1],b_target[2][2],b_target[3][0],b_target[3][1],b_target[3][2]))
    msg = sensor_msgs.msg.JointState()
    msg.header = std_msgs.msg.Header()
    msg.header.stamp = rospy.Time.now()
    msg.name = [body_part+'_axis0', body_part+'_axis1', body_part+'_axis2']
    msg.velocity = [0,0,0]
    msg.effort = [0,0,0]
    euler = [res.x[0]/180*math.pi,res.x[1]/180*math.pi,res.x[2]/180*math.pi]
    if body_part=="head":
        msg.position = [-euler[0], euler[2], euler[1]]
    elif body_part=="wrist_left":
        msg.position = [-euler[1], -euler[0], -euler[2]]
    elif body_part=="shoulder_left":
        msg.position = [euler[1], euler[0], euler[2]]
    else:
        msg.position = [euler[0], euler[1], euler[2]]
    joint_state.publish(msg)
    pos_estimate_prev = res.x

rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, magneticsCallback,queue_size=1)
rospy.spin()
