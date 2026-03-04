import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh, duplicate_node_pairs
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time

# Caida libre
na = 23; nb = 19
np.random.seed(10)
X, T = createRectangularMesh(a = 1, b = 1, na = na, nb = nb, h = 0.75)
X[:,2] = (1 - np.exp(3*(X[:,1]-0.5)))*X[:,2]

#make copy
Y = X.copy(); 
Y[:,2] = -Y[:,2]
X = np.concatenate([X,Y]); T = np.concatenate([T,T+na*nb])

seam = duplicate_node_pairs(X)
print(seam.shape)
X[:,2] += 1.8; 
#X += 0.0002*np.random.randn(X.shape[0],3) 


self = Cloth(X, T, seam); 
dt = self.estimateTimeStep(L=2)
self.setSimulatorParameters(dt = dt, thck = 0.99, mu_s = 0.35, str = 0.001*1e-4, shr = 2.5*1e-4, 
                            tol = 0.0075, kappa = 1.5*1e-4, kappa_bnd = 0.5*1e-4,  mu_f = 0.25)
print(self.corners)
self.plotMesh()

tf = int(3/dt); t = np.linspace(0,2*np.pi,tf); freq = 2
inds_ctr = [0]
u = X[inds_ctr]

start_time = time.time()
for i in range(tf):
    self.simulate(u = u, control = inds_ctr)
u = self.positions[inds_ctr]; 
for i in range(0*tf):
    u[:,1] += 0.0065*np.sin(freq*t[i])
    self.simulate(u = u, control = inds_ctr)
inds_ctr = []
u = self.positions[inds_ctr]; 
for i in range(tf):
    self.simulate(u = u, control = inds_ctr)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))

self.makeMovie(speed=5,repeat=False,smooth=2)
