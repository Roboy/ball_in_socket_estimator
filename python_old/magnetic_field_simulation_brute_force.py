import numpy as np
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
import math, os, time
import geometry_msgs
from multiprocessing import Pool, freeze_support
from pathlib import Path

num_processes =60
num_random_samples = 1000
show_plot = False
b_target = [(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
# define sensor
sensor_pos = [[-22.7,7.7,0],[-14.7,-19.4,0],[14.7,-19.4,0],[22.7,7.7,0]]#[[22.7,7.7,0],[14.7,-19.4,0],[-14.7,-19.4,0],[-22.7,7.7,0]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    s = Sensor(pos=pos,angle=90,axis=(0,0,1))
    sensors.append(s)
# def gen_magnets():
#     return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

cs = 4
dimx,dimy,dimz = 5,5,5

field_strength = 1000
# hallbach 0, works well
def gen_magnets(mag_pos):
    magnets = []
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                if mag_pos[i*dimx+j*dimy+k*dimz]==1:
                    magnets.append(Box(mag=(field_strength,0,0),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
                if mag_pos[i*dimx+j*dimy+k*dimz]==2:
                    magnets.append(Box(mag=(-field_strength,0,0),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
                if mag_pos[i*dimx+j*dimy+k*dimz]==3:
                    magnets.append(Box(mag=(0,field_strength,0),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
                if mag_pos[i*dimx+j*dimy+k*dimz]==4:
                    magnets.append(Box(mag=(0,-field_strength,0),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
                if mag_pos[i*dimx+j*dimy+k*dimz]==5:
                    magnets.append(Box(mag=(0,0,field_strength),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
                if mag_pos[i*dimx+j*dimy+k*dimz]==6:
                    magnets.append(Box(mag=(0,0,-field_strength),dim=(cs,cs,cs),pos=((i-dimx//2)*(cs+1),(j-dimy//2)*(cs+1),(k-dimz//2)*(cs+1))))
    return magnets

def func(mp):
    start_time = time.time()
    # print(mp)
    angle_error_sum = 0
    b_field_error_sum = 0
    for axis in range(0,3):
        for iter in np.linspace(-80,80,6):
            c = Collection(gen_magnets(mp))
            if axis==0:
                rot = [iter,0,0]
            if axis==1:
                rot = [0,iter,0]
            if axis==2:
                rot = [0,0,iter]
            c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
            c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
            c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
            b_target = []
            for sens in sensors:
                b_target.append(sens.getB(c))

            def min_func_wrapper(mp):
                def min_func(x):
                    c = Collection(gen_magnets(mp))
                    c.rotate(x[0],(1,0,0), anchor=(0,0,0))
                    c.rotate(x[1],(0,1,0), anchor=(0,0,0))
                    c.rotate(x[2],(0,0,1), anchor=(0,0,0))
                    b_error = 0
                    i = 0
                    for sens in sensors:
                        b_error = b_error + np.linalg.norm(sens.getB(c)-b_target[i])
                        i=i+1
                    # print(b_error)
                    return [b_error,b_error,b_error]
                return min_func

            res = least_squares(min_func_wrapper(mp), [0,0,0], bounds = ((-90,-90,-90), (90, 90, 90)))
            angle_error = ((rot[0]-res.x[0])**2+(rot[1]-res.x[1])**2+(rot[2]-res.x[2])**2)**0.5
            b_field_error = res.cost
            # print("target %.3f %.3f %.3f\tresult %.3f %.3f %.3f\tb-field error %.3f\tangle_error %.3f"%(rot[0],rot[1],rot[2],res.x[0],res.x[1],res.x[2],b_field_error,angle_error))
            angle_error_sum+=angle_error
            b_field_error_sum+=b_field_error
            if angle_error_sum>10:
                elapsed_time = time.time() - start_time
                print("failed after %ds (%.3f,%.3f,%.3f), angle_error %.3f b_field_error %.3f"%(elapsed_time,rot[0],rot[1],rot[2],angle_error_sum,b_field_error_sum))
                return [mp,angle_error_sum+1000000,b_field_error_sum]
    elapsed_time = time.time() - start_time
    print("success after %ds, angle_error %.3f b_field_error %.3f"%(elapsed_time,angle_error_sum,b_field_error_sum))
    return [mp,angle_error_sum,b_field_error_sum]

if show_plot:
    # calculate B-field on a grid
    xs = np.linspace(-40,40,33)
    ys = np.linspace(-40,40,44)
    zs = np.linspace(-40,40,44)
    POS0 = np.array([(x,0,z) for z in zs for x in xs])
    POS1 = np.array([(x,y,0) for y in ys for x in xs])

    for i in range(0,dimx*dimy*dimz):
        mag_pos[i] = random.randint(0, 6)

    rot = [0,0,0]

    c = Collection(gen_magnets(mag_pos))
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))

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
    displaySystem(c, subplotAx=ax1, sensors=sensors, suppress=True, direc=True)
    plt.show()

# result = func(mag_pos)
# print(result)

print("starting poseestimator")
args = []
for i in range(0,num_random_samples):
    mag_pos = np.zeros(dimx*dimy*dimz)
    for j in range(0,dimx*dimy*dimz):
        mag_pos[j] = random.randint(0, 6)
    args.append(mag_pos)
print(args)

with Pool(processes=num_processes) as pool:
    results = pool.starmap(func, zip(args))
    print(results)
    min_index = 0
    min_value = 1e10
    i=0
    for res in results:
        if res[1]<min_value:
            min_index = i
            min_value = results[i][1]
        i+=1
    print("best result %d with angle_error %.3f:"%(min_index, min_value))
    print(results[min_index][0])
