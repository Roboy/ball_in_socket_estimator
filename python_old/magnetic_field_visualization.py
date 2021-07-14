import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math

iterations = 1000000

# define sensor
sensor_pos = [[-22.7,7.7,0],[-14.7,-19.4,0],[14.7,-19.4,0],[22.7,7.7,0]]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    sensors.append(Sensor(pos=pos))

def gen_magnets():
    return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

# calculate B-field on a grid
xs = np.linspace(-40,40,33)
ys = np.linspace(-40,40,44)
zs = np.linspace(-40,40,44)
POS0 = np.array([(x,0,z) for z in zs for x in xs])
POS1 = np.array([(x,y,0) for y in ys for x in xs])

body_part = "test"

record = open("/home/letrend/workspace/roboy3/"+body_part+"_data0.log","w")
record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")

first = True

for iter in range(iterations):
    rot = [random.uniform(-50,50),random.uniform(-50,50),random.uniform(-90,90)]

    c = Collection(gen_magnets())
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))

    data = []
    for sens in sensors:
        data.append(sens.getB(c))

    record.write(str(data[0][0])+ " " + str(data[0][1]) + " " + str(data[0][2])+ " " + str(data[1][0])+ " " + str(data[1][1])+ " " + str(data[1][2])+ " " + str(data[2][0])+ " " + str(data[2][1])+ " " + str(data[2][2])  + " " + str(data[3][0])+ " " + str(data[3][1])+ " " + str(data[3][2])+ " " + str(rot[0]/180.0*math.pi) + " " + str(rot[1]/180.0*math.pi) + " " + str(rot[2]/180.0*math.pi) + "\n")

    if first:
        # create figure
        fig = plt.figure(figsize=(18,7))
        ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(132)                   # 2D-axis
        ax3 = fig.add_subplot(133)                   # 2D-axis
        Bs = c.getB(POS0).reshape(44,33,3)     #<--VECTORIZED
        X,Y = np.meshgrid(xs,ys)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

        Bs = c.getB(POS1).reshape(44,33,3)     #<--VECTORIZED
        X,Z = np.meshgrid(xs,zs)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
        sensor_visualization = []
        i = 0
        for pos in sensor_pos:
             sensor_visualization.append(Box(mag=(0,0,0.001),dim=(1,1,1),pos=sensor_pos[i]))
             i = i+1
        d = Collection(c,sensor_visualization)
        displaySystem(d, subplotAx=ax1, suppress=True)
        plt.show()
        first = False
    if iter%10==0:
        print("(%d/%d)"%(iter,iterations))
record.close()
