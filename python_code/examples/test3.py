import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import time

# Caida libre
n = 28; na = n; nb = n
m = np.int32(np.floor(n/2))
np.random.seed(1)
X, T = createRectangularMesh(a = 1, b = 1, na = na, nb = nb, h = 0.2)
X[:,2] += 1.25; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
dt = self.estimateTimeStep(L=1)
self.setSimulatorParameters(thck=0.95,mu_s=0.2,dt=dt,mu_f = 0.2, kappa = 1.1*1e-4, tol = 0.0075, delta = 0.09)
self.plotMesh()
tf = int(6/dt)
inds = [409]; u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    #print(i)
    if i == int(tf/2):
        inds = [na-1]
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 5, repeat = True, smooth = 2)
#kernprof -l -v test3.py > perfil_selfcols3.txt