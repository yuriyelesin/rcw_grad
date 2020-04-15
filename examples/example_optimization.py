#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os,sys
Nthread = 4
os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

import autograd.numpy as np
from autograd import grad
import nlopt, numpy as npf
import matplotlib.pyplot as plt

rpath = '/Users/weiliang/Documents/rcw_grad'
sys.path.append(rpath)
sys.path.append("../") 

import use_autograd
use_autograd.use = 1
import rcwa


# In[14]:


nG = 101 # truncation order, the actual truncation order might differ from this
# frequency and angles
freq = 1.
theta = 0.
phi = 0.
# lattice vector
Lx = .5
Ly = .5
L1 = [Lx,0.]
L2 = [0.,Ly]

# now consider 4 layers: vacuum + patterned + patterned + vacuum
epsuniform0 = 1. # dielectric for layer 1 (uniform)
epsuniformN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickN = 1.  # thickness for vacuum layer N

###########.  patterned for optimization
# all patterned layers below have the same griding structure: Nx*Ny
Nx = 100
Ny = 100
Nlayer = 2  # number of patterned layers for optimization
ndof = Nx*Ny*Nlayer # total number of DOFs

# thickness
thickness = [0.5,0.5]
epsbkg = [1., 1.]
epsdiff = [3.,5.]


# In[15]:


ctrl = 0
vec = []
def fun_reflection(dof,Qabs):
    freqcmp = freq*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    # add all layers in order
    obj.Add_LayerUniform(thick0,epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(thickness[i],epsdiff[i],epsbkg[i],Nx,Ny)
    obj.Add_LayerUniform(thick0,epsuniformN)
    
    obj.Init_Setup(Gmethod=0)

    p_amp = 1.
    p_phase = 0.
    s_amp = 0.
    s_phase = 0.
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    obj.GridLayer_getDOF(dof)
    R,_ = obj.RT_Solve(normalize=1)


    if 'autograd' not in str(type(R)):
        global ctrl
        global vec
        vec = npf.copy(dof)
        
        print(ctrl,R)
        if npf.mod(ctrl,5)==0:
            for i in range(Nlayer):
                plt.figure();
                plt.imshow(np.reshape(dof[i*Nx*Ny:(i+1)*Nx*Ny],(Nx,Ny)))
                plt.colorbar()
                plt.show()
            
        ctrl +=1
    return R


# In[16]:


Qabs = 20.
fun = lambda dof: fun_reflection(dof,Qabs)
grad_fun = grad(fun)
def fun_nlopt(dof,gradn):
    gradn[:] = grad_fun(dof)
    return fun(dof)

init = np.random.random(ndof)
lb=np.zeros(ndof,dtype=float)
ub=np.ones(ndof,dtype=float)

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_max_objective(fun_nlopt)
x = opt.optimize(init)


# In[ ]:




