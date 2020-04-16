#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os,sys
# Nthread = 1
# os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import pi
import numpy as np
from matplotlib.collections import PatchCollection
sys.path.append("../") 
# change it to 1 when using autograd
import use_autograd
use_autograd.use = 0
import rcwa

def GetAngleForMode(power):
    return 0

# magma
def PlotModePower(G, power, rmin = 0.05, rmax = 0.5, cmap = 'cividis'):
    n = len(G)
    xmin = np.min(G[:,0]) - 1
    xmax = np.max(G[:,0]) + 1
    ymin = np.min(G[:,1]) - 1
    ymax = np.max(G[:,1]) + 1
    
    pmin = np.min(power)
    pmax = np.max(power)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # Create sizes for the circles
    sizes = []
    for i in range(n):
        sizes.append(rmin + (rmax - rmin)/(pmax - pmin)*power[i])

    circles = [plt.Circle(G[i], radius=sizes[i]) for i in range(n)]
    col = PatchCollection(circles, array=np.array(power), cmap=cmap)
    ax.add_collection(col)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.colorbar(col)
    plt.show()

# In[3]:


nG = 144 # truncation order, the actual truncation order might differ from this
Gmethod = 0
# lattice vector
Lx = .1
Ly = .1
L1 = [Lx,0.]
L2 = [0.,Ly]

# all patterned layers below have the same griding structure: Nx*Ny
Nx = 150
Ny = 150

# frequency and angles
freq = 19.9
theta = 0.0#np.pi/6
phi = np.pi*0

p_amp = 1.
s_amp = 0.
p_phase = 0.
s_phase = 0.

# now consider 4 layers: vacuum + patterned + patterned + vacuum
epsuniform0 = 1. # dielectric for layer 1 (uniform)
epsuniformN = 1.  # dielectric for layer N (uniform)


pattern_layer_num = 3
thick0 = 1. # thickness for vacuum layer 1
thickp = []
for l in range(pattern_layer_num):
    thickp.append(0.1)
thickN = 1.  # thickness for vacuum layer N

# for patterned layer, eps = epsbkg + dof * epsdiff
epsbkg = 1.
epsdiff = 12.+1j*0

# setup RCWA
obj = rcwa.RCWA_obj(nG,L1,L2,freq,theta,phi)
obj.Add_LayerUniform(thick0,epsuniform0)
for l in range(pattern_layer_num):
    obj.Add_LayerGrid(thickp[l],epsdiff,epsbkg,Nx,Ny)
obj.Add_LayerUniform(thickN,epsuniformN)
obj.Init_Setup(Gmethod=Gmethod)


# In[4]:


# Now set up epsilon of patterned layers: epsilon = epsbkg + dof * epsdiff
dof = []
# set up grid-1
radius = 0.1
dof.append(np.zeros((Nx,Ny)))
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
dof[-1][sphere]=1

# set up grid-2
radius = 0.2
dof.append(np.zeros((Nx,Ny)))
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
dof[-1][sphere]=1

# set up grid-3
radius = 0.3
dof.append(np.zeros((Nx,Ny)))
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
dof[-1][sphere]=1


if 0:
    plt.figure();
    plt.imshow(dof[0])
    plt.colorbar()
    plt.show()
     
    plt.figure();
    plt.imshow(dof[1])
    plt.colorbar()
    plt.show()
     
    plt.figure();
    plt.imshow(dof[2])
    plt.colorbar()
    plt.show()


# the total dof passing to rcwa will be concatenating all dofs in layer order, so the length will be Nx*Ny*NPatternedlayer
dofs = np.concatenate((dof[0].flatten(), dof[1].flatten(), dof[2].flatten()))

# Now add DOF to rcwa
obj.GridLayer_getDOF(dofs.flatten())

obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0, direction = 'forward')
# R,T= obj.RT_Solve(normalize=1)
# print("R:",R,", T:",T,", A:",1-R-T)

R,T= obj.RT_SolveComponents(normalize=1)
RR = []
TT = []
for g in range(obj.nG):
    RR.append(R[g] + R[g+obj.nG])
    TT.append(T[g] + T[g+obj.nG])
# print(np.min(RR), np.max(RR), np.min(TT), np.max(TT))
total_power_r = np.sum(np.abs(RR))
total_power_t = np.sum(np.abs(TT))
print("R:",total_power_r,", T:",total_power_t,", A:",1-total_power_r-total_power_t)

# ai, bi = obj.GetAmplitudes(4, z_offset=1.0)
# power_t = []
# for g in range(obj.nG):
#     power_t.append(np.real(ai[g]*np.conj(ai[g]) + ai[g+obj.nG]*np.conj(ai[g+obj.nG])))
# power_t = np.array(power_t)*obj.normalization
# total_power_t = np.sum(power_t)
#  
#  
# ai, bi = obj.GetAmplitudes(0, z_offset=0.0)
# power_r = []
# for g in range(obj.nG):
#     power_r.append(np.real(bi[g]*np.conj(bi[g]) + bi[g+obj.nG]*np.conj(bi[g+obj.nG])))
# power_r = np.array(power_r)*obj.normalization
# total_power_r = np.sum(power_r)
# print("R:",total_power_r,", T:",total_power_t,", A:",1-total_power_r-total_power_t)


PlotModePower(obj.G, TT)
    

# print(obj.G)





# In[ ]:




