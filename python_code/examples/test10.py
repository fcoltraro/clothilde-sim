import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")
from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh, duplicate_node_pairs, weld_quad_mesh
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
import time

points_df = pd.read_csv("meshes/Samarreta3D_punts.csv")
quads_df = pd.read_csv("meshes/Samarreta3D_quads.csv")

X = points_df[['x', 'y', 'z']].to_numpy(dtype=float)
T = quads_df[['v0', 'v1', 'v2', 'v3']].to_numpy(dtype=int)
X[:,2] += 0.75
seam = duplicate_node_pairs(X , tol=1e-4)


self = Cloth(X, T,seam); 
dt = 1/60
self.setSimulatorParameters(dt = dt, thck = 1.15, mu_s = 0.2, str = 0.01*1e-4, shr = 25*1e-4,
                            tol = 0.0075, kappa = 0.075*1e-4, kappa_bnd = 0.0075*1e-4,  mu_f = 0.15, sub_steps = 8)
self.plotMesh()


tf = int(4/dt); 
print(tf)
inds_ctr = [0,1]
u = X[inds_ctr]

start_time = time.time()
for i in range(tf):
    self.simulate(u = u, control = inds_ctr)
tf = int(4/dt); 
inds_ctr = []
u = self.positions[inds_ctr]; 
for i in range(tf):
    self.simulate(u = u, control = inds_ctr)

print('Time:',time.time()-start_time)
print('Average iterations',self.total_iters/(len(self.history_pos)-1))

self.makeMovie(speed=1,repeat=True,smooth=2)