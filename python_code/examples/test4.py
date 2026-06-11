import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
import time

# Caida libre
na = 7; nb = 7
m = np.int32(np.floor(na/2))
np.random.seed(1)
X, T = createRectangularMesh(a = 0.6, b = 0.6, na = na, nb = nb, h = 0.1)
X[:,2] += 0.4; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
dt = 1/600
self.setSimulatorParameters(dt=dt,tol=0.0085, 
                            rho=0.1,delta=0.1,kappa=0.25*1e-4,shr=0.5*1e-4, kappa_bnd = 0.025*1e-4,
                            str=0.001*1e-4,alpha=0.2,mu_f=0.3,mu_s=0.3,thck=0.9,sub_steps=1)
tf = int(0.65/dt)
print(tf)
self.plotMesh()
inds = [m-1, nb*na - m-1]
u = self.positions[inds]
start_time = time.time()
for i in range(275):
    print("Iteration: ",i)
    if i == -int(tf/2):
        inds = [0]
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 5, repeat = False, smooth = 0)
#kernprof -l -v test4.py > perfil_selfcols4.txt