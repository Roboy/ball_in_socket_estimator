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
import pandas
from pandas import DataFrame
from pandas import concat
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, get_context, set_start_method

num_processes = 62

if len(sys.argv) < 4:
    print("\nUSAGE: python3 magnetic_field_collision.py dataset_name ball_joint_config multi_processing, e.g. \n python3 magnetic_field_collision.py head_data0.log two_magnets.yaml 0\n")
    sys.exit()

dataset_name = sys.argv[1]
balljoint_config = load(open(sys.argv[2], 'r'), Loader=Loader)
multi_processing = sys.argv[3]=='1'

dataset = pandas.read_csv('/home/letrend/workspace/roboy3/'+dataset, delim_whitespace=True)
dataset = dataset.values[0:len(dataset),0:]

sensor_values = []
sensor_values.append(dataset[:,0:3])
sensor_values.append(dataset[:,3:6])
sensor_values.append(dataset[:,6:9])
sensor_values.append(dataset[:,9:12])
pos = dataset[:,12:]
print('first sensor values')
print(sensor_values[0][0])
print(sensor_values[1][0])
print(sensor_values[2][0])
print(sensor_values[3][0])
print('first ball position')
print(pos[0])

number_of_samples = len(pos)
print('number of samples %d'%number_of_samples)
position_difference_sensitivity = 1/180*math.pi
magnetic_field_difference_sensitivity = 1.44

comparisons = (number_of_samples-1)*number_of_samples/2
print('comparisons %d'%comparisons)
print('approx time %d seconds or %f minutes'%(comparisons/1283370,comparisons/1283370/60))
timestamp = time.strftime("%H:%M:%S")
print('start time: %s'%timestamp)
collisions = 0
if multi_processing:
    def collisionFunc(i):
        collision_j = []
        magnetic_field_difference = 0
        # position_difference = []
        global iterations
        collisions = 0
        for j in range(i+1,number_of_samples):
            mag_diff = 0
            for k in range(0,4):
                mag_diff += np.linalg.norm(sensor_values[k][i]-sensor_values[k][j])
            magnetic_field_difference+=mag_diff
            # pos_diff = np.linalg.norm(pos[i]-pos[j])
            # position_difference.append(pos_diff)
            if mag_diff<magnetic_field_difference_sensitivity:# and pos_diff>position_difference_sensitivity:
                collisions+=1
                collision_j.append(j)
                if(collisions%100==0):
                    print("collisions: %d"%collisions)

        return (collisions,i,collision_j,magnetic_field_difference)
    args = range(0,number_of_samples,1)
    with Pool(processes=num_processes) as pool:
        start = time.time()
        results = pool.starmap(collisionFunc, zip(args))
        end = time.time()
        collisions = 0
        magnetic_field_difference = 0
        for n in range(0,number_of_samples):
            collisions += results[n][0]
            magnetic_field_difference += results[n][3]
            for j in results[n][2]:
                print("%d--%d"%(n,j))
                print(pos[n])
                print(pos[j])
                print("---")



    print('comparisons: %d'%comparisons)
    print('collisions: %d'%collisions)
    print('average magnetic_field_difference: %f'%(magnetic_field_difference/comparisons))
    print('actual time: %d'%(end - start))


    # print(magnetic_field_difference)
    # print(position_difference)
else:
    status = tqdm(total=comparisons, desc='comparisons', position=0)

    iter = 0
    for i in range(0,number_of_samples):
        for j in range(i+1,number_of_samples):
            mag_diff = 0
            for k in range(0,4):
                mag_diff += np.linalg.norm(sensor_values[k][i]-sensor_values[k][j])
            # magnetic_field_difference.append(mag_diff)
            pos_diff = np.linalg.norm(pos[i]-pos[j])
            # position_difference.append(pos_diff)
            if mag_diff<magnetic_field_difference_sensitivity and pos_diff>position_difference_sensitivity:
                collisions+=1
                p0 = pos[i]
                p1 = pos[j]
                if(collisions%100==0):
                    print("collisions: %d"%collisions)
                # for k in range(0,4):
                #     print(k)
                #     print(sensor_values[k][i])
                #     print(sensor_values[k][j])
                # print('positions in degree')
                # print(pos[i]*180/math.pi)
                # print(pos[j]*180/math.pi)
                # print('mag_diff')
                # print(mag_diff)
                # print('pos_diff in degree')
                # print(pos_diff*180/math.pi)
                # print('oh oh')

            iter+=1
            status.update(1)


    print('comparisons: %d'%comparisons)
    print('collisions: %d'%collisions)
    # print(magnetic_field_difference)
    # print(position_difference)
