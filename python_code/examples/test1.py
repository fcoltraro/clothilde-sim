import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth0 import Cloth
from implementation.utils import createRectangularMesh, colored_grid_edges
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time

# Caida libre
na = 30; nb = 30
np.random.seed(1)
X, T = createRectangularMesh(a = 1, b = 1, na = na, nb = nb, h = 0.2)
edge_sets = colored_grid_edges(na, nb)
X[:,2] += 1; 
X += 0.0001*np.random.randn(X.shape[0],3) 

self = Cloth(X, T, sets = edge_sets); 
dt = 0.3*self.estimateTimeStep(L=1)
self.setSimulatorParameters(dt = dt, thck = 1, mu_s = 0.4, tol = 0.01, alpha = 0.3,
                            str=0.05*1e-4,shr = 0.25*1e-4, kappa=0.25*1e-4, kappa_bnd = 0.01*1e-4)
self.plotMesh()

tf = int(6/dt)
inds = [0,na-1]
u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    print(i)
    if i == int(tf/2):
        inds = []
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 12, repeat = True, smooth = 2)
#self.plotMesh()
#self.saveFrames(speed = 4)

#kernprof -l -v test1.py > perfil_selfcols.txt