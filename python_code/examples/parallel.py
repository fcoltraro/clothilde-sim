import sys,os
notebook_dir = os.getcwd()  # Gets current working directory
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")
from implementation.Cloth import Cloth 
from implementation.utils import createRectangularMesh
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time

# Caida libre
n = 20; na = n; nb = n
X, T = createRectangularMesh(a = 0.7, b = 0.7, na = na, nb = nb, h = 0.25)
X[:,2] += 0.75; tf = int(3/(1/60)); 

def run_one_cloth(args):
    j, X, T, na, tf = args
    
    cloth = Cloth(X, T)
    cloth.setSimulatorParameters(dt=1/60, tol=0.005,
                                 rho=0.1, delta=0.1, kappa=0.5*1e-4,
                                 shr=20*1e-4, str=0.05*1e-4, alpha=0.3,
                                 mu_f=0.25, mu_s=0.35)
    #simulation loop
    inds = [0]
    for k in range(tf):
        cloth.simulate(u=cloth.positions[inds], control=inds)
    #remove CHOLMOD factors to make the object picklable (for playback only)
    #alternative: just return positions and velocities 
    del cloth.factor_E
    del cloth.factor_Et
    for stretch in cloth.stretch:
        stretch.factor = None
    for shear in cloth.shear:
        shear.factor = None
    
    return j, cloth  

if __name__ == "__main__":
    n_envs = 10
    args_list = [(j, X, T, na, tf) for j in range(n_envs)]
    classes = [None] * n_envs

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_envs) as ex:
        for j, cloth_obj in ex.map(run_one_cloth, args_list):
            print("Finished cloth", j)
            classes[j] = cloth_obj

    print("Time:", time.time() - start_time)

    classes[3].makeMovie(1, True, 2)

