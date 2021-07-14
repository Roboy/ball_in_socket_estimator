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
    print("\nUSAGE: python3 magnetic_config_printout.py balljoint_config, e.g. \n python3 magnetic_config_printout.py balljoint_config.yaml \n")
    sys.exit()
balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print("id: %d"%balljoint_config['id'])
print("calibration")
balljoint_config['calibration']
print("field_strength")
print(balljoint_config['field_strength'])
print("sensor_pos_offsets")
for offset in balljoint_config['sensor_pos_offsets']:
    print(offset)
print("sensor_angle_offsets")
for offset in balljoint_config['sensor_angle_offsets']:
    print(offset)
print("magnet_pos_offsets")
for offset in balljoint_config['magnet_pos_offsets']:
    print(offset)
print("magnet_angle_offsets")
for offset in balljoint_config['magnet_angle_offsets']:
    print(offset)
