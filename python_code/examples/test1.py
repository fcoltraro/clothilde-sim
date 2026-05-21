import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time

# Caida libre
na = 30; nb = 30
np.random.seed(1)
X, T = createRectangularMesh(a = 0.8, b = 0.8, na = na, nb = nb, h = 0.1)
X[:,2] += 0.9; 
X += 0.0001*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
dt = 1/60 
self.plotMesh()
self.setSimulatorParameters(dt = dt, thck = 0.95, mu_s = 0.3, tol = 0.0075, shr = 5*1e-4, 
                            kappa=0.25*1e-4, kappa_bnd = 0.025*1e-4, str = 0.005*1e-4, sub_steps=9,slf=0.0025)

tf = int(6/dt)
inds = [0,na-1]
u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    if i == int(tf/2):
        inds = []
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 1, repeat = False, smooth = 2)
#self.plotMesh()
#self.saveFrames(speed = 4)

#kernprof -l -v test1.py > perfil_selfcols.txt