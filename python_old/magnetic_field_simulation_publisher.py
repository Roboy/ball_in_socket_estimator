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

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_field_simulation_publisher.py balljoint_config_yaml sensor_log_file, e.g. \n python3 magnetic_field_simulation_publisher.py test.yaml sensor_log.yaml\n")
    sys.exit()

balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
sensor_log_file = sys.argv[2]

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
rospy.init_node('magnetic_field_simulation_publisher')
magneticSensor_pub = rospy.Publisher('roboy/middleware/MagneticSensor', MagneticSensor, queue_size=1)

if(path.exists(sensor_log_file)):
    sensor_log = load(open(sensor_log_file, 'r'), Loader=Loader)
else:
    sensor_log = {'position':[],'sensor_values':[]}

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
    plt.show()

magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
plotMagnets(magnets)

def JointTarget(data):
    rospy.loginfo("new joint target: %f %f %f"%(data.position[0],data.position[1],data.position[2]))
    magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                    balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
    magnets.rotate(angle=data.position[0]*180.0/math.pi,axis = (1,0,0), anchor=(0,0,0))
    magnets.rotate(angle=data.position[1]*180.0/math.pi,axis = (0,1,0), anchor=(0,0,0))
    magnets.rotate(angle=data.position[2]*180.0/math.pi,axis = (0,0,1), anchor=(0,0,0))
    msg = MagneticSensor()
    msg.id = int(balljoint_config['id'])
    values = []
    for sens in sensors:
        val = sens.getB(magnets)
        msg.x.append(val[0])
        msg.y.append(val[1])
        msg.z.append(val[2])
        values.append((val[0],val[1],val[2]))
    magneticSensor_pub.publish(msg)
    # sensor_log['position'].append((data.position[0]*180.0/math.pi,data.position[1]*180.0/math.pi,data.position[2]*180.0/math.pi))
    # sensor_log['sensor_values'].append(values)
    # print(sensor_log)

joint_state_sub = rospy.Subscriber('joint_targets', sensor_msgs.msg.JointState, JointTarget, queue_size=1)

rospy.spin()

rospy.loginfo('writing sensor log to file %s'%sensor_log_file)
with open(sensor_log_file, 'w') as file:
    documents = dump(sensor_log, file, default_flow_style=False)

rospy.loginfo('done')
