import numpy as np
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
import math

# define sensor
sensor_pos = [(-22.7,7.7,0),(-14.7,-19.4,0),(14.7,-19.4,0),(22.7,7.7,0)]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    sensors.append(Sensor(pos=pos))

def gen_magnets():
    return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

mlab.options.offscreen = True
fig = mlab.figure(bgcolor=(1,1,1), size=(1500, 1500), fgcolor=(0, 0, 0))
for iter in range(126,360):
    c = Collection(gen_magnets())
    c.rotate(iter,(0,1,0),(0,0,0))
    x_lower,x_upper = -20, 20
    y_lower,y_upper = -20, 20
    z_lower,z_upper = -20, 20
    grid_spacing_value = 1#0.25
    xd = int((x_upper-x_lower)/grid_spacing_value)
    yd = int((y_upper-y_lower)/grid_spacing_value)
    zd = int((z_upper-z_lower)/grid_spacing_value)

    x, y, z = np.mgrid[x_lower:x_upper:xd*1j,
                      y_lower:y_upper:yd*1j,
                      z_lower:z_upper:zd*1j]

    xs = np.linspace(x_lower,x_upper,xd)
    ys = np.linspace(y_lower,y_upper,yd)
    zs = np.linspace(z_lower,z_upper,zd)

    U = np.zeros((xd,yd,zd))
    V = np.zeros((xd,yd,zd))
    W = np.zeros((xd,yd,zd))
    i = 0
    print("simulating magnetic data")
    for zi in zs:
        POS = np.array([(x,y,zi) for x in xs for y in ys])
        Bs = c.getB(POS).reshape(len(xs),len(ys),3)
        U[:,:,i]=Bs[:,:,0]
        V[:,:,i]=Bs[:,:,1]
        W[:,:,i]=Bs[:,:,2]
        i=i+1
        print("(%d/%d)"%(i,zd))
        # print(Bs)

    i = 0
    print("generating flow")

    fig.scene.disable_render = True
    for xi in xs:
        st = mlab.flow(x, y, z, U,V,W, line_width=0.1,
                              seedtype='plane', integration_direction='both',figure=fig,opacity=0.05)
        st.streamline_type = 'tube'
        st.seed.visible = False
        st.tube_filter.radius = 0.1
        st.seed.widget.origin = np.array([ xi, y_lower,  z_upper])
        st.seed.widget.point1 = np.array([ xi, y_upper,  z_upper])
        st.seed.widget.point2 = np.array([ xi, y_lower,  z_lower])
        st.seed.widget.resolution = 10#int(xs.shape[0])
        st.seed.widget.enabled = True
        st.seed.widget.handle_property.opacity = 0
        st.seed.widget.plane_property.opacity = 0
        st.update_pipeline()
        print("(%d/%d)"%(i,zd))
        i= i+1
        mlab.show(stop=True)
    fig.scene.disable_render = False
    mayavi.mlab.view(azimuth=iter/4, elevation=70)
    mlab.axes(extent = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper],figure=fig)
    mlab.savefig('/home/letrend/Videos/magnetic_arrangements/[500,0,0]_[0,500,0]_[0,0,500]/movie001/'+'anim%05d.png'%(iter))
    mlab.clf()
    del U,V,W


# fig.scene.movie_maker.record = True
# fig.scene.movie_maker.directory = '/home/letrend/Videos/magnetic_arrangements/[500,0,0]_[0,500,0]_[0,0,500]'
# view = mayavi.mlab.view()
#
# @mlab.animate(delay=10, ui=True)
# def anim():
#     for i in range(360):
#         mayavi.mlab.view(azimuth=i, elevation=70)
#         yield
# anim()
# mlab.show()
# print("done")

