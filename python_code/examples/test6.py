import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir)
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import time

na = 35; nb = 20
X, T = createRectangularMesh(a = 0.9,b = 0.5, na = na, nb = nb, h = 0.05)
X[:,2] += 0.6; #adjust height

clothilde = Cloth(X, T)

# solver parameters
dt = 0.002 #time step
tol = 0.0095 # up to 0.75% of relative error in constraint satisfaction to stop iterations

#physical parameters
rho = 0.1 #cloth density
delta = 0.08 # aerodynamics parameter: between 0 and rho
kappa = 0.25*1e-4 # stifness or bending resistance
kappa_bnd = 0.01*1e-4 # stifness or bending resistance
alpha = 0.2 #damping of oscillations
shr = 0.5*1e-4 #allowed shearing resistance
str = 0.005*1e-4 #allowed stretching resistance
mu_f = 0.45 #friction with the floor
mu_s = 0.35 #friction with the cloth itself
thck = 0.95 #size of the balls

clothilde.setSimulatorParameters(dt=dt,tol=tol,
                                rho=rho,delta=delta,kappa=kappa,kappa_bnd = kappa_bnd, shr=shr,
                                str=str,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

tf = 1000; t = np.linspace(0,2*np.pi,tf); freq = 3
inds_ctr = [0,na-1]
u = X[inds_ctr]

start_time = time.time()
for i in range(1000):
    clothilde.simulate(u = u, control = inds_ctr)
u = clothilde.positions[inds_ctr]; 
for i in range(tf):
    u[:,1] += 0.0035*np.sin(freq*t[i])
    clothilde.simulate(u = u, control = inds_ctr)
inds_ctr = [0]
u = clothilde.positions[inds_ctr]; 
for i in range(1000):
    clothilde.simulate(u = u, control = inds_ctr)

print('Time:',time.time()-start_time)
print('Average iterations',clothilde.total_iters/(len(clothilde.history_pos)-1))


clothilde.makeMovie(5,True,2)

#kernprof -l -v test6.py > perfil_selfcols6.txt