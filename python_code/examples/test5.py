import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import quad_cylinder_mesh
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time

np.random.seed(1)
X, T = quad_cylinder_mesh(R=0.11, H=0.22, h=0.016,f=1)

X[:,2] += 0.35; 
X += 0.0002*np.random.randn(X.shape[0],3) 
self = Cloth(X,T); 
dt = self.estimateTimeStep(L=2*np.pi*0.11)
dt = 0.002
self.plotMesh()
self.setSimulatorParameters(dt = dt, delta= 0.1, kappa= 0.2*1e-4, tol = 0.0075, str = 0.0025*1e-4, shr = 2*1e-4, mu_s = 0.35, thck = 0.95, mu_f = 0.25)



tf = int(5/dt)

inds = [np.argmax(X[:, 2] + X[:, 1])]; 
u = self.positions[inds]
start_time = time.time()
for i in range(tf):
    #print("Iteration: ",i)
    if i == int(tf/2):
        inds = []
        u = self.positions[inds]
    self.simulate(u = u, control = inds)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))



self.makeMovie(speed=4,repeat=True,smooth=2)
#kernprof -l -v test7.py > perfil_selfcols7.txt

