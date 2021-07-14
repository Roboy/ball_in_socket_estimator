import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random
import math, os, time, sys
import geometry_msgs
from multiprocessing import Pool, freeze_support
from pathlib import Path
from yaml import load,dump,Loader,Dumper

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_field_calibration.py balljoint_config_yaml sensor_log_file, e.g. \n python3 magnetic_field_calibration.py test.yaml sensor_log.yaml\n")
    sys.exit()

class MagneticFieldCalibrator:
    balljoint_config = None
    sensor_log = None
    number_of_magnets = 0
    number_of_sensors = 0
    number_of_parameters = 0
    initial = []
    upper_bound = []
    lower_bound = []
    calib = []
    config_file = None

    def __init__(self, ball_joint_config_file, sensor_log_file):
        self.balljoint_config = load(open(ball_joint_config_file, 'r'), Loader=Loader)
        print(self.balljoint_config)
        self.sensor_log = load(open(sensor_log_file, 'r'), Loader=Loader)
        print(self.sensor_log)
        self.config_file = ball_joint_config_file
        self.number_of_magnets = len(self.balljoint_config['field_strength'])
        self.number_of_sensors = len(self.balljoint_config['sensor_pos'])

        print("optimizing: ")
        for c in self.balljoint_config['calibration']['optimize']:
            if c=='field_strength':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets
                for i in range(0,self.number_of_magnets):
                    self.initial.append(1300)
                    self.upper_bound.append(1600)
                    self.lower_bound.append(1000)
                self.calib.append(0)
                print('\tfield_strength')
            if c=='magnet_pos':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets*3
                for i in range(0,self.number_of_magnets*3):
                    self.initial.append(0)
                    self.upper_bound.append(5)
                    self.lower_bound.append(-5)
                self.calib.append(1)
                print('\tmagnet_pos')
            if c=='magnet_angle':
                self.number_of_parameters = self.number_of_parameters+self.number_of_magnets*3
                for i in range(0,self.number_of_magnets*3):
                    self.initial.append(0)
                    self.upper_bound.append(10)
                    self.lower_bound.append(-10)
                self.calib.append(2)
                print('\tmagnet_angle')
            if c=='sensor_pos':
                self.number_of_parameters = self.number_of_parameters+self.number_of_sensors*3
                for i in range(0,self.number_of_sensors*3):
                    self.initial.append(0)
                    self.upper_bound.append(5)
                    self.lower_bound.append(-5)
                self.calib.append(3)
                print('\tsensor_pos')
            if c=='sensor_angle':
                self.number_of_parameters = self.number_of_parameters+self.number_of_sensors*3
                for i in range(0,self.number_of_sensors*3):
                    self.initial.append(0)
                    self.upper_bound.append(10)
                    self.lower_bound.append(-10)
                self.calib.append(4)
                print('\tsensor_angle')

        print('number_of_magnets: %d\nnumber_of_sensors: %d\nnumber_of_parameters: %d'\
            %(self.number_of_magnets,self.number_of_sensors,self.number_of_parameters))
        # self.visualizeSetup()
        self.optimize()

    def gen_sensors(self,pos,pos_offset,angle,angle_offset):
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

    def gen_magnets(self,field_strength,pos,pos_offset,angle,angle_offset):
        magnets = []
        i = 0
        for field in field_strength:
            magnet = Box(mag=(0,0,field), \
             dim=(self.balljoint_config['magnet_dimension'][i][0],self.balljoint_config['magnet_dimension'][i][1],self.balljoint_config['magnet_dimension'][i][2]),\
             pos=(pos[i][0]+pos_offset[i][0],pos[i][1]+pos_offset[i][1],pos[i][2]+pos_offset[i][2]))
            magnet.rotate(angle=angle[i][0]+angle_offset[i][0],axis=(1,0,0))
            magnet.rotate(angle=angle[i][1]+angle_offset[i][1],axis=(0,1,0))
            magnet.rotate(angle=angle[i][2]+angle_offset[i][2],axis=(0,0,1))
            magnets.append(magnet)
            i = i+1
        return magnets

    def decode(self,x):
        field_strength = []
        magnet_pos_offsets = []
        magnet_angle_offsets = []
        sensor_pos_offsets = []
        sensor_angle_offsets = []

        for i in range(0,self.number_of_magnets):
            field_strength.append(1300)
            magnet_pos_offsets.append([0,0,0])
            magnet_angle_offsets.append([0,0,0])
        for i in range(0,self.number_of_sensors):
            sensor_pos_offsets.append([0,0,0])
            sensor_angle_offsets.append([0,0,0])
        j = 0
        for c in self.calib:
            if c==0:
                for i in range(0,self.number_of_magnets):
                    field_strength[i] = x[j]
                    j = j+1
            if c==1:
                for i in range(0,self.number_of_magnets):
                    magnet_pos_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==2:
                for i in range(0,self.number_of_magnets):
                    magnet_angle_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==3:
                for i in range(0,self.number_of_sensors):
                    sensor_pos_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
            if c==4:
                for i in range(0,self.number_of_sensors):
                    sensor_angle_offsets[i] = [x[j],x[j+1],x[j+2]]
                    j = j+3
        return field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets

    def func(self, x):
        magnet_pos = self.balljoint_config['magnet_pos']
        magnet_angle = self.balljoint_config['magnet_angle']

        sensor_pos = self.balljoint_config['sensor_pos']
        sensor_angle = self.balljoint_config['sensor_angle']

        field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets = self.decode(x)

        sensors = self.gen_sensors(sensor_pos,sensor_pos_offsets,sensor_angle,sensor_angle_offsets)

        b_error = 0
        j =0
        for pos in self.sensor_log['position']:
            c = Collection(self.gen_magnets(field_strength,magnet_pos,magnet_pos_offsets,\
                magnet_angle,magnet_angle_offsets))
            c.rotate(angle=pos[0]*180.0/math.pi,axis=(1,0,0), anchor=(0,0,0))
            c.rotate(angle=pos[1]*180.0/math.pi,axis=(0,1,0), anchor=(0,0,0))
            c.rotate(angle=pos[2]*180.0/math.pi,axis=(0,0,1), anchor=(0,0,0))
            # print("%f %f %f"%(pos[0],pos[1],pos[2]))
            i = 0
            for sens in sensors:
                b_error = b_error + np.linalg.norm(sens.getB(c)-self.sensor_log['sensor_values'][j][i])
                i=i+1
            j =j +1
        # print(b_error)
        return [b_error]

    def optimize(self):
        res = least_squares(self.func, self.initial,\
            bounds = (self.lower_bound, self.upper_bound),\
            ftol=1e-8, \
            xtol=1e-8,verbose=2,\
            max_nfev=self.balljoint_config['calibration']['max_nfev'])

        field_strength = self.balljoint_config['field_strength']
        magnet_pos = self.balljoint_config['magnet_pos']
        magnet_angle = self.balljoint_config['magnet_angle']

        sensor_pos = self.balljoint_config['sensor_pos']
        sensor_angle = self.balljoint_config['sensor_angle']

        field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets = self.decode(res.x)

        sensors = self.gen_sensors(sensor_pos,sensor_pos_offsets,sensor_angle,sensor_angle_offsets)

        print("b_field_error with calibration: %f\n"%self.func(res.x)[0])

        j = 0
        pos = self.sensor_log['position']
        for target in self.sensor_log['sensor_values']:
            print("target b_field for %f %f %f"%(pos[j][0],pos[j][1],pos[j][2]))
            i = 0
            for sens in sensors:
                print('%.4f    %.4f    %.4f'%(target[i][0],target[i][1],target[i][2]))
                i = i+1
            print("b_field with calibration:")
            c = Collection(self.gen_magnets(field_strength,magnet_pos,magnet_pos_offsets, \
                                            magnet_angle,magnet_angle_offsets))
            c.rotate(angle=pos[j][0]*180.0/math.pi,axis=(1,0,0), anchor=(0,0,0))
            c.rotate(angle=pos[j][1]*180.0/math.pi,axis=(0,1,0), anchor=(0,0,0))
            c.rotate(angle=pos[j][2]*180.0/math.pi,axis=(0,0,1), anchor=(0,0,0))
            for sens in sensors:
                mag = sens.getB(c)
                print('%.4f    %.4f    %.4f'%(mag[0],mag[1],mag[2]))
            print('----------------------------------')
            j = j+1

        print("\noptimization results:\n")
        for c in self.calib:
            if c==0:
                print('field_strength')
                print(field_strength)
                self.balljoint_config['field_strength'] = field_strength
            if c==1:
                print('magnet_pos_offsets')
                print(magnet_pos_offsets)
                self.balljoint_config['magnet_pos_offsets'] = magnet_pos_offsets
            if c==2:
                print('magnet_angle_offsets')
                print(magnet_angle_offsets)
                self.balljoint_config['magnet_angle_offsets'] = magnet_angle_offsets
            if c==3:
                print('sensor_pos_offsets')
                print(sensor_pos_offsets)
                self.balljoint_config['sensor_pos_offsets'] = sensor_pos_offsets
            if c==4:
                print('sensor_angle_offsets')
                print(sensor_angle_offsets)
                self.balljoint_config['sensor_angle_offsets'] = sensor_angle_offsets

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        input("Enter to write optimization to config file %s_%s..."%(timestamp,self.config_file))

        with open(timestamp+'_'+self.config_file, 'w') as file:
            documents = dump(self.balljoint_config, file)

    def visualizeSetup(self):
        field_strength = self.balljoint_config['field_strength']
        magnet_pos = self.balljoint_config['magnet_pos']
        magnet_angle = self.balljoint_config['magnet_angle']

        sensor_pos = self.balljoint_config['sensor_pos']
        sensor_angle = self.balljoint_config['sensor_angle']

        field_strength, magnet_pos_offsets, magnet_angle_offsets, sensor_pos_offsets, sensor_angle_offsets = self.decode(self.initial)

        for pos in self.sensor_log['position']:

            magnets = Collection(self.gen_magnets(field_strength,magnet_pos,magnet_pos_offsets, \
                                            magnet_angle,magnet_angle_offsets))
            magnets.rotate(angle=pos[0]*180.0/math.pi,axis=(1,0,0), anchor=(0,0,0))
            magnets.rotate(angle=pos[1]*180.0/math.pi,axis=(0,1,0), anchor=(0,0,0))
            magnets.rotate(angle=pos[2]*180.0/math.pi,axis=(0,0,1), anchor=(0,0,0))

            sensors = self.gen_sensors(sensor_pos,sensor_pos_offsets,sensor_angle,sensor_angle_offsets)

            # calculate B-field on a grid
            xs = np.linspace(-40,40,33)
            ys = np.linspace(-40,40,44)
            zs = np.linspace(-40,40,44)
            POS0 = np.array([(x,0,z) for z in zs for x in xs])
            POS1 = np.array([(x,y,0) for y in ys for x in xs])

            fig = plt.figure(figsize=(18,7))
            ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
            ax2 = fig.add_subplot(132)                   # 2D-axis
            ax3 = fig.add_subplot(133)                   # 2D-axis
            Bs = magnets.getB(POS0).reshape(44,33,3)     #<--VECTORIZED
            X,Y = np.meshgrid(xs,ys)
            U,V = Bs[:,:,0], Bs[:,:,2]
            ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

            Bs = magnets.getB(POS1).reshape(44,33,3)     #<--VECTORIZED
            X,Z = np.meshgrid(xs,zs)
            U,V = Bs[:,:,0], Bs[:,:,2]
            ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
            displaySystem(magnets, subplotAx=ax1, suppress=True, sensors=sensors, direc=True)

            for sens in sensors:
                print(sens.getB(magnets))

            plt.show()

MagneticFieldCalibrator(sys.argv[1],sys.argv[2])
