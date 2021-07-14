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
import sys, time
from os import path

if len(sys.argv) < 2:
    print("\nUSAGE: python3 magnetic_config_merge.py config_1 config_2 ... result_config, e.g. \
        \n python3 magnetic_config_merge.py magnet_A.yaml magnet_B.yaml magnet_C.yaml balljoint_config.yaml \n")
    sys.exit()
config = []
result = load(open(sys.argv[len(sys.argv)-1], 'r'), Loader=Loader)

i=0
for offset in result['sensor_pos_offsets']:
    j = 0
    for val in offset:
        result['sensor_pos_offsets'][i][j] = 0
        j = j+1
i=0
for offset in result['sensor_angle_offsets']:
    j = 0
    for val in offset:
        result['sensor_angle_offsets'][i][j] = 0
        j = j+1
i=0
for offset in result['magnet_pos_offsets']:
    j = 0
    for val in offset:
        result['magnet_pos_offsets'][i][j] = 0
        j = j+1
i=0
for offset in result['magnet_angle_offsets']:
    j = 0
    for val in offset:
        result['magnet_angle_offsets'][i][j] = 0
        j = j+1

for i in range(1,len(sys.argv)-1):
    config.append(load(open(sys.argv[i], 'r'), Loader=Loader))
    result['field_strength'][i-1] = config[i-1]['field_strength'][0]
    k = 0
    for offset in config[i-1]['sensor_pos_offsets']:
        print(offset)
        j = 0
        for val in offset:
            result['sensor_pos_offsets'][k][j] += val
            j = j+1
        k+=1
    print("sensor_angle_offsets")
    k = 0
    for offset in config[i-1]['sensor_angle_offsets']:
        print(offset)
        j = 0
        for val in offset:
            result['sensor_angle_offsets'][k][j] += val
            j = j+1
        k+=1
    print("magnet_pos_offsets")
    k = 0
    for offset in config[i-1]['magnet_pos_offsets']:
        print(offset)
        j = 0
        for val in offset:
            result['magnet_pos_offsets'][k][j] += val
            j = j+1
        k+=1
    print("magnet_angle_offsets")
    k = 0
    for offset in config[i-1]['magnet_angle_offsets']:
        print(offset)
        j = 0
        for val in offset:
            result['magnet_angle_offsets'][k][j] += val
            j = j+1
        k+=1

i=0
for offset in result['sensor_pos_offsets']:
    j = 0
    for val in offset:
        result['sensor_pos_offsets'][i][j] = val/len(config)
        j = j+1
    i+=1
i=0
for offset in result['sensor_angle_offsets']:
    j = 0
    for val in offset:
        result['sensor_angle_offsets'][i][j] = val/len(config)
        j = j+1
    i+=1
i=0
for offset in result['magnet_pos_offsets']:
    j = 0
    for val in offset:
        result['magnet_pos_offsets'][i][j] = val/len(config)
        j = j+1
    i+=1
i=0
for offset in result['magnet_angle_offsets']:
    j = 0
    for val in offset:
        result['magnet_angle_offsets'][i][j] = val/len(config)
        j = j+1
    i+=1

print(result)

timestamp = time.strftime("%Y%m%d-%H%M%S")
input("Enter to write to config file %s_%s..."%(timestamp,sys.argv[-1]))

with open(timestamp+'_'+sys.argv[-1], 'w') as file:
    documents = dump(result, file)
