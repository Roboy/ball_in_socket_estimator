#!/usr/bin/python3
import magjoint
import sys, time
import numpy as np

if len(sys.argv) < 2:
    print("\nUSAGE: ./magnetic_collision_cuda.py ball_joint_config visualize_only, e.g. \n python3 magnetic_collision_cuda.py two_magnets.yaml 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
visualize_only = sys.argv[2]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets(ball.config)
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

print('\n----------------first course search\n')
grid_positions = []
for i in np.arange(-80,80,5):
    for j in np.arange(-80,80,5):
        for k in np.arange(-80,80,5):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
start = time.time()
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools

mod = SourceModule("""
  __global__ void distance(int number_of_samples, float3 *p1, float3 *p2, float3 *p3, float3 *p4, float *d)
  {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i>=number_of_samples || j>=number_of_samples || j<i)
        return;
    d[i*number_of_samples+j] = sqrtf(powf(p1[i].x-p1[j].x,2.0) + powf(p1[i].y-p1[j].y,2.0) + powf(p1[i].z-p1[j].z,2.0)) +
                               sqrtf(powf(p2[i].x-p2[j].x,2.0) + powf(p2[i].y-p2[j].y,2.0) + powf(p2[i].z-p2[j].z,2.0)) +
                               sqrtf(powf(p3[i].x-p3[j].x,2.0) + powf(p3[i].y-p3[j].y,2.0) + powf(p3[i].z-p3[j].z,2.0)) +
                               sqrtf(powf(p4[i].x-p4[j].x,2.0) + powf(p4[i].y-p4[j].y,2.0) + powf(p4[i].z-p4[j].z,2.0));
  };
  """)

distance = mod.get_function("distance")

number_of_samples = len(sensor_values)
p1 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p2 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p3 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p4 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
i = 0
for val in sensor_values:
    p1[i] = val[0]
    p2[i] = val[1]
    p3[i] = val[2]
    p4[i] = val[3]
    i = i+1
print(p1)
p1_gpu = gpuarray.to_gpu(p1)
p2_gpu = gpuarray.to_gpu(p2)
p3_gpu = gpuarray.to_gpu(p3)
p4_gpu = gpuarray.to_gpu(p4)
comparisons = int(((number_of_samples-1)*(number_of_samples/2)))
out_gpu = gpuarray.empty(number_of_samples**2, np.float32)
print(out_gpu)
print('calculating %d collisions'%comparisons)
number_of_samples = np.int32(number_of_samples)

bdim = (16, 16, 1)
dx, mx = divmod(number_of_samples, bdim[0])
dy, my = divmod(number_of_samples, bdim[1])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))))
print(bdim)
print(gdim)
distance(number_of_samples, p1_gpu, p2_gpu, p3_gpu, p4_gpu, out_gpu, block=bdim, grid=gdim)
end = time.time()
print('took: %d s or %f min'%(end - start,(end - start)/60))
out = np.reshape(out_gpu.get(),(number_of_samples,number_of_samples))
print(out)
