import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.ClothBary import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import time

# Caida libre
n = 30; na = n; nb = n
m = np.int32(np.floor(n/2))
np.random.seed(10)
X,Q,T = createRectangularMesh(a = 1, b = 1, na = na, nb = nb, h = 0.1)
X[:,2] += 1.25; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X,Q,T); 
dt = 1/60
self.setSimulatorParameters(dt = dt, thck = 1.1, mu_s = 0.3, str = 0.005*1e-4, kappa_bnd = 0.075*1e-4, 
                            shr = 1.5*1e-4, tol = 0.0075, kappa = 1.2*1e-4, mu_f = 0.3, sub_steps = 10, slf = 0*0.005)
self.plotMesh()
tf = int(3.0/dt)
inds = [643]; u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    self.simulate(u = u, control = inds)
tf = int(2.5/dt)
t = np.linspace(0,2*np.pi,tf)
inds = [na-1]
u0 = u = self.positions[inds]
for j in range(tf):    
    print(j)
    u[:,2] = u0[:,2] + 0.02*np.sin(2*t[j])
    self.simulate(u = u, control = inds)
tf = int(4/dt)
inds = []
for k in range(tf):
    u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 1, repeat = False, smooth = 2)
#kernprof -l -v test3.py > perfil_selfcols3.txt