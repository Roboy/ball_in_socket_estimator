import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math
from multiprocessing import Pool, freeze_support, get_context, set_start_method
from yaml import load,dump,Loader,Dumper
import sys
import rospy

num_processes = 60
iterations = 1000
grid_x_min = -60
grid_x_max = 60
grid_y_min = -60
grid_y_max = 60
grid_z_min = -80
grid_z_max = 80
grid_step_x = 1
grid_step_y = 1
grid_step_z = 1

if len(sys.argv) < 7:
    print("\nUSAGE: python3 training_data_generation.py balljoint_config_yaml body_part normalize_magnetic_strength sampling_method number_of_samples visualize_only , e.g. \n python3 training_data_generation.py test.yaml head 0 random 100000 0\n")
    sys.exit()

balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
body_part = sys.argv[2]
normalize_magnetic_strength = sys.argv[3]=='1'
sampling_method = sys.argv[4]
iterations = int(sys.argv[5])
visualize_only = sys.argv[6]=='1'

if(sampling_method=='random'):
    print('generating %d sampled'%iterations)

if normalize_magnetic_strength:
    rospy.logwarn("normalizing magnetic field")
else:
    rospy.logwarn("NOT! normalizing magnetic field")

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

def plotMagnets(magnets):
    # calculate B-field on a grid
    xs = np.linspace(-40,40,33)
    ys = np.linspace(-40,40,44)
    zs = np.linspace(-40,40,44)
    POS0 = np.array([(x,0,z) for z in zs for x in xs])
    POS1 = np.array([(x,y,0) for y in ys for x in xs])

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

sensors = gen_sensors(balljoint_config['sensor_pos'],balljoint_config['sensor_pos_offsets'],balljoint_config['sensor_angle'],balljoint_config['sensor_angle_offsets'])

grid_position = []

def generateMagneticData(iter):
    if sampling_method=='random':
        if body_part=="wrist_left":
            rot = [random.uniform(-40,40),random.uniform(-40,40),random.uniform(-40,40)]
        elif body_part=="head":
            rot = [random.uniform(-50,50),random.uniform(-50,50),random.uniform(-90,90)]
        elif body_part=="shoulder_left":
            rot = [random.uniform(-70,70),random.uniform(-70,70),random.uniform(-45,45)]
        else:
            rot = [random.uniform(-90,90),random.uniform(-90,90),random.uniform(-90,90)]
    elif sampling_method=='grid':
        rot = grid_position[iter]
    # rot = [0,0,0]

    magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                     balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))
    magnets.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    magnets.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    magnets.rotate(rot[2],(0,0,1), anchor=(0,0,0))
    data = []
    for sens in sensors:
        val = sens.getB(magnets)
        if normalize_magnetic_strength:
            val /= np.linalg.norm(val)
        data.append(val)
    if(iter%1000==0):
        print("%d/%d"%(iter,iterations))
    return (data, rot)

magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_dimension'],balljoint_config['magnet_pos'],balljoint_config['magnet_pos_offsets'], \
                                balljoint_config['magnet_angle'],balljoint_config['magnet_angle_offsets']))

data_norm = []
data = []
for sens in sensors:
    val0 = sens.getB(magnets)
    val1 = sens.getB(magnets)
    data.append(val0)
    val1 /= np.linalg.norm(val1)
    data_norm.append(val1)
print("sensor values:\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n"%(data[0][0],data[0][1],data[0][2],data[1][0],data[1][1],data[1][2],data[2][0],data[2][1],data[2][2],data[3][0],data[3][1],data[3][2]))
print("sensor values normalized:\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n"%(data_norm[0][0],data_norm[0][1],data_norm[0][2],data_norm[1][0],data_norm[1][1],data_norm[1][2],data_norm[2][0],data_norm[2][1],data_norm[2][2],data_norm[3][0],data_norm[3][1],data_norm[3][2]))

if visualize_only:
    plotMagnets(magnets)
    sys.exit()

record = open("/home/letrend/workspace/roboy3/"+body_part+"_data0.log","w")
record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")

if sampling_method=='grid':
    iterations = 0
    for i in np.arange(grid_x_min,grid_x_max,grid_step_x):
        for j in np.arange(grid_y_min,grid_y_max,grid_step_y):
            for k in np.arange(grid_z_min,grid_z_max,grid_step_z):
                grid_position.append([i,j,k])
                iterations+=1

set_start_method('fork',True)
args = range(0,iterations,1)
with Pool(processes=num_processes) as pool:
    results = pool.starmap(generateMagneticData, zip(args))
    for i in range(0,iterations):
        if(i%10000==0):
            print("%d/%d"%(i,iterations))
        record.write(\
            str(results[i][0][0][0])+ " " + str(results[i][0][0][1]) + " " + str(results[i][0][0][2])+ " " + \
            str(results[i][0][1][0])+ " " + str(results[i][0][1][1])+ " " + str(results[i][0][1][2])+ " " + \
            str(results[i][0][2][0])+ " " + str(results[i][0][2][1])+ " " + str(results[i][0][2][2])  + " " + \
            str(results[i][0][3][0])+ " " + str(results[i][0][3][1])+ " " + str(results[i][0][3][2])+ " " + \
            str(results[i][1][0]/180.0*math.pi) + " " + str(results[i][1][1]/180.0*math.pi) + " " + str(results[i][1][2]/180.0*math.pi) + "\n")
record.close()
print('data saved to /home/letrend/workspace/roboy3/'+body_part+'_data0.log')
