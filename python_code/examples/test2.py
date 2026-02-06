import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
import time

# Caida libre
n = 27; na = n; nb = n
np.random.seed(1)
X, T = createRectangularMesh(a = 0.7, b = 0.7, na = na, nb = nb, h = 0.1)
X[:,2] += 0.35; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
dt = self.estimateTimeStep(L=0.7)

self.setSimulatorParameters(shr=3*1e-4, dt = 0.0025, tol = 0.01)
self.plotMesh()
tf = 2000
inds = [0, nb*na - 1]
start_time = time.time()
for i in range(tf):
    if i == 1000:
        inds = []
    self.simulate(u = X[inds], control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))

self.makeMovie(speed = 5, repeat = True, smooth = 2)
#kernprof -l -v test2.py > perfil_selfcols2.txt