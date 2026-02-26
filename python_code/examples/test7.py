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
n = 30; na = n; nb = n
m = np.int32(np.floor(n/2))
np.random.seed(10)
X, T = createRectangularMesh(a = 1, b = 1, na = na, nb = nb, h = 0.01)
X[:,2] += 1.25; 

X += 0.0002*np.random.randn(X.shape[0],3) 

self = Cloth(X, T); 
dt = 0.002# self.estimateTimeStep(L=1)
self.setSimulatorParameters(dt = dt, thck = 1.1, mu_s = 0.4, str = 1*1e-4, shr = 20*1e-4, tol = 0.005, kappa = 0.5*1e-4)
self.plotMesh()
inds = [0,na-1,na*(nb-1),na*nb -1]; u = self.positions[inds]
start_time = time.time()
tf = int(2/dt)
pa = self.positions[inds[0]]
pb = self.positions[inds[1]]
c = 0.5*(pa + pb)
r = 0.5*np.linalg.norm(pa-pb)
t = np.linspace(0,2*np.pi,tf)
w = 1
u0 = self.positions[inds]
for j in range(tf):
    #print("iteration :",j)
    #u = self.positions[inds]
    u[0,:] = c + r*np.array([-np.cos(w*t[j]),0,np.sin(w*t[j])])
    u[1,:] = c - r*np.array([-np.cos(w*t[j]),0,np.sin(w*t[j])])
    u[2,:] = u0[2,:]; u[3,:] = u0[3,:]
    self.simulate(u = u, control = inds)
tf = int(0.5/dt)
u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))


self.makeMovie(speed = 5, repeat = False, smooth = 2)
#kernprof -l -v test3.py > perfil_selfcols3.txt