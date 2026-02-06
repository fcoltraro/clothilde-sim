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
n = 10; na = 3*n; nb = 2*n
m = np.int32(np.floor(n/2))
np.random.seed(1)
X, T = createRectangularMesh(a = 0.6, b = 0.4, na = na, nb = nb, h = 0.15)
X[:,2] += 0.5; 

X += 0.0001*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
self.setSimulatorParameters(shr=0.5*1e-4,str=0.005*1e-4,kappa=0.05*1e-4,dt=0.0025)

print(self)
self.plotMesh()
tf = 2000
inds = [0,na-1]
u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    print("Iteration: ",i)
    if i == -500:
        inds = [0,na-1]
        u = self.positions[inds]
    if i == -1000:
        inds = []
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Inextensibility iters:',self.total_inxt/(len(self.history_pos)-1))
print('Collisions iters:',self.total_cols/(len(self.history_pos)-1))

self.makeMovie(2,True,2)
#kernprof -l -v test5.py > perfil_selfcols5.txt