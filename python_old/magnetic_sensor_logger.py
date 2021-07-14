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
from tqdm import tqdm
average_samples = 100

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_sensor_logger.py balljoint_config sensor_log_file, e.g. \n python3 magnetic_sensor_logger.py balljoint_config.yaml sensor_log.yaml\n")
    sys.exit()
balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print(dump(balljoint_config, Dumper=Dumper))
sensor_log_file = sys.argv[2]
if(path.exists(sensor_log_file)):
    sensor_log = load(open(sensor_log_file, 'r'), Loader=Loader)
else:
    sensor_log = {'position':[],'sensor_values':[]}

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

def plotMagnets(magnets):
    # calculate B-field on a grid
    xs = np.linspace(-40,40,33)
    ys = np.linspace(-40,40,44)
    zs = np.linspace(-40,40,44)
    POS0 = np.array([(x,0,z) for z in zs for x in xs])
    POS1 = np.array([(x,y,0.9) for y in ys for x in xs])

    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
    ax2 = fig.add_subplot(133)                   # 2D-axis
    ax3 = fig.add_subplot(132)                   # 2D-axis

    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    Bs = magnets.getB(POS0).reshape(44,33,3)     #<--VECTORIZED
    X,Y = np.meshgrid(xs,zs)
    U,V = Bs[:,:,0], Bs[:,:,2]
    ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    Bs = magnets.getB(POS1).reshape(44,33,3)     #<--VECTORIZED
    X,Z = np.meshgrid(xs,ys)
    U,V = Bs[:,:,0], Bs[:,:,1]
    ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
    displaySystem(magnets, subplotAx=ax1, suppress=True, sensors=sensors, direc=True)

    for sens in sensors:
        print(sens.getB(magnets))

    plt.show()

ball_pos = [[0,0,0],[0,0,(math.pi/4)],[0,0,(math.pi/4)*2],[0,0,(math.pi/4)*3],[0,0,(math.pi/4)*4],[0,0,(math.pi/4)*5],[0,0,(math.pi/4)*6],[0,0,(math.pi/4)*7],[math.pi/2,0,0],[-math.pi/2,0,0]]

magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
print(ball_pos[0])
magnets.rotate(ball_pos[0][0]*180/math.pi,(1,0,0), anchor=(0,0,0))
magnets.rotate(ball_pos[0][1]*180/math.pi,(0,1,0), anchor=(0,0,0))
magnets.rotate(ball_pos[0][2]*180/math.pi,(0,0,1), anchor=(0,0,0))
plotMagnets(magnets)

k = 0
j = 0
sensor_value_counter = 0
values = []
pbar = tqdm(total=average_samples)

def MagneticSensorCallback(data):
    global k
    global j
    global pbar
    global sensor_value_counter
    global values

    if k>=len(ball_pos):
        rospy.signal_shutdown("no more")
        return

    if data.id is not int(balljoint_config['id']):
        return
    sensor_value_counter = sensor_value_counter+1

    if sensor_value_counter<10:
        pbar = tqdm(total=average_samples)
        j = 0
        i =0
        values = []
        for sens in data.x:
            values.append([data.x[i],data.y[i],data.z[i]])
            i = i+1
        return

    if j<average_samples:
        pbar.update(1)
        i =0
        for sens in data.x:
            values[i][0]=values[i][0]+data.x[i]
            values[i][1]=values[i][1]+data.y[i]
            values[i][2]=values[i][2]+data.z[i]
            i = i+1
        j = j+1
        return
    j = 0
    i=0
    for sens in data.x:
        values[i][0]=values[i][0]/average_samples
        values[i][1]=values[i][1]/average_samples
        values[i][2]=values[i][2]/average_samples
        i = i+1
    sensor_value_counter = 0
    sensor_log['position'].append(ball_pos[k])
    sensor_log['sensor_values'].append(values)

    print(sensor_log)

    magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                    balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
    magnets.rotate(ball_pos[k][0]*180/math.pi,(1,0,0), anchor=(0,0,0))
    magnets.rotate(ball_pos[k][1]*180/math.pi,(0,1,0), anchor=(0,0,0))
    magnets.rotate(ball_pos[k][2]*180/math.pi,(0,0,1), anchor=(0,0,0))
    print('simulated:')
    for sens in sensors:
        print(sens.getB(magnets))
    print('recorded:')
    for v in values:
        print(v)
    k = k+1
    input("Press Enter for next ball position: %f %f %f"%(ball_pos[k][0],ball_pos[k][1],ball_pos[k][2]))

rospy.init_node('magnetic_sensor_logger')
magneticSensor_sub = rospy.Subscriber('roboy/middleware/MagneticSensor', MagneticSensor, MagneticSensorCallback, queue_size=1)

rospy.spin()

rospy.loginfo('writing sensor log to file %s'%sensor_log_file)
with open(sensor_log_file, 'w') as file:
    documents = dump(sensor_log, file, default_flow_style=False)

rospy.loginfo('done')
