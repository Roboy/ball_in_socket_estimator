import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math
import sensor_msgs.msg, std_msgs
from yaml import load,dump,Loader,Dumper
import sys
from os import path

if len(sys.argv) < 2:
    print("\nUSAGE: python3 magnetic_sensor_selector.py balljoint_config, e.g. \n python3 magnetic_sensor_logger.py balljoint_config.yaml \n")
    sys.exit()
balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print('listening to sensor with id %d'%balljoint_config['id'])

def MagneticSensorCallback(data):
    if data.id is not int(balljoint_config['id']):
        return

    values = []
    i =0
    print("sensor_id=%d  --------------------\n"%data.id)
    for sens in data.x:
        print('%.3f\t%.3f\t%.3f'%(data.x[i],data.y[i],data.z[i]))
        i = i+1
    print("\n---------------------------------\n")
    rospy.sleep(0.1)
rospy.init_node('magnetic_sensor_selector')
magneticSensor_sub = rospy.Subscriber('roboy/middleware/MagneticSensor', MagneticSensor, MagneticSensorCallback, queue_size=1)

rospy.spin()

rospy.loginfo('done')
