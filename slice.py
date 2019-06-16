import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
from mpl_toolkits.axes_grid1 import AxesGrid
def PyGadget(gadget_path):
    gadget_file = open(gadget_path,'rb')
    gadget_file.seek(0)
    header_len = np.int(np.fromfile(gadget_file,dtype=np.int32,count=1).copy())
    #header_len=256
    #print header_len
    gadget_file.seek(4)
    particle_num = np.fromfile(gadget_file,dtype=np.int32,count=6).copy()
    #print particle_num
    gadget_file.seek(4+4*6)
    particle_mass = np.fromfile(gadget_file,dtype=np.double,count=6).copy()
    #print particle_mass
    time = np.fromfile(gadget_file,dtype=np.double,count=1).copy()
    redshift = np.fromfile(gadget_file,dtype=np.double,count=1).copy()
    gadget_file.seek(4+header_len+4+4)
    particle_pos_data = np.fromfile(gadget_file,dtype=np.float32,count=3*np.int(particle_num[1])).copy()
    gadget_file.seek(4+header_len+4+4+np.int(particle_num[1])*3*4+4+4)
    particle_vel_data = np.fromfile(gadget_file,dtype=np.float32,count=3*np.int(particle_num[1])).copy()
    if particle_mass[1]==0.0:
        gadget_file.seek(4+header_len+4+4+np.int(particle_num[1])*3*4+4+4+np.int(particle_num[1])*3*4+4+4+np.int(particle_num[1])*4+4+4)
        particle_mass_data = np.fromfile(gadget_file,dtype=np.float32,count=np.int(particle_num[1])).copy()
        particle_mass = particle_mass_data.reshape([np.int(particle_num[1]),1])
    particle_pos = particle_pos_data.reshape([np.int(particle_num[1]),3])
    particle_vel = particle_vel_data.reshape([np.int(particle_num[1]),3])
    gadget_file.close()
    del gadget_file,header_len,particle_num
    return particle_pos,particle_vel,particle_mass[1],time,redshift

def denslice(filename,defilename,sign):
    p,v,m,t,z=PyGadget(filename)
    x=np.transpose(p)[0]
    y=np.transpose(p)[1]
    z=np.transpose(p)[2]
    fil=(z<110000) & (z>100000)
    fil2=np.random.rand(len(fil))<0.1

    detable=np.loadtxt(defilename,delimiter=',',unpack=True)
    loga=detable[0][1:]
    logk=np.transpose(detable)[0][1:]
    logdedk=np.transpose(np.transpose(detable[1:])[1:])
    itpdedk=itp.interp2d(loga,logk,logdedk)

    des=np.histogram2d(x[fil],y[fil],bins=np.linspace(0,400000,256))
    desk=np.fft.fft2(des[0])
    dedesk=np.copy(desk)

    unittrans=(400.0/256.0)**2.
    for i in np.linspace(0,254,255,dtype=int):
        for j in np.linspace(0,254,255,dtype=int):
            if i<127:
                kx=i+0.5
            if i>=127:
                kx=254.5-i
            if j<127:
                ky=j+0.5
            if j>=127:
                ky=254.5-j
            k2 = kx**2.+ky**2.
            phylogk = np.log10(4.0 * np.pi * np.sqrt(k2) / ( 0.001 * 400000.0 ))
            dedesk[i][j]=sign*dedesk[i][j]*(10.**itpdedk(np.log10(t),phylogk))
    dedes=np.fft.ifft2(dedesk)
    return np.transpose(des[0])*m/unittrans,np.transpose(np.real(dedes))

