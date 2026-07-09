import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.ClothQuads import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
import time

# Caida libre
n = 25; na = n; nb = n
np.random.seed(1)
X, Q, T = createRectangularMesh(a = 0.7, b = 0.7, na = na, nb = nb, h = 0.1)
X[:,2] += 0.4; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X, Q, T); 
dt = 1/60

self.setSimulatorParameters(shr=1*1e-4, dt = dt, tol = 0.005, thck = 0.9, kappa=0.25*1e-4, mu_f=0.25, kappa_bnd = 0, slf=1e-4, sub_steps=8, mu_s= 0.4)
self.plotMesh()
tf = int(5/dt)
inds = [0,na-1]
start_time = time.time()
for i in range(tf):
    if i == int(tf/2):
        inds = []
    self.simulate(u = X[inds], control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))

self.makeMovie(speed = 1, repeat = True, smooth = 1)
#kernprof -l -v test2.py > perfil_selfcols2.txt