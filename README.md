# clothilde-sim: accurate physical simulation of (quasi)-inextensible textiles 


https://github.com/user-attachments/assets/00d62006-e6ad-4f0a-8ace-b711f61eee81


This repository contains a cloth simulator specifically designed for **robotics and control** tasks. The simulator prioritizes:

* _Inelastic limit_: the simulator is able to simulate (quasi)-inextensible fabrics efficiently.
* _Aerodynamic effects_: interaction with the air can be incorporated (even in the absence of wind).
* _Stability and robustness_: no jittering and stable contact and friction behaviour. 
* _Physical consistency_: the simulator is very stable under various remeshings of the cloth.
* _Easy of use and modularity_: just a few lines of Python code are enough to start simulating.

The simulator has been used and validated in the context of **dynamic textile manipulation** by robots, showcasing its realism when compared with real recordings of various textiles. For more information on robotic applications and validations, click <a href="https://fcoltraro.github.io/projects/telas/"> here</a>.


https://github.com/user-attachments/assets/4dcf4124-d486-414f-9cc4-a79dca4d09b4

---

## 1. Installation

We recomend using the miniconda package and a conda enviroment for simulation (especially in Mac and Windows systems). 

### 1.1 Requirements

The simulator is natively implemented in Python (>=3.11) and relies on the following computing libraries: Numpy, Scipy, scikit-sparse, CHOLMOD and pykdtree. Visualization is done by Polyscope and profiling by line_profiler.

### 1.2 Installation Steps

#### Step 0: Clone the repository 
```
git clone https://github.com/fcoltraro/clothilde-sim.git
cd clothilde
```
#### Step 1: Create a new conda environment with the required packages
```
conda create -n clothilde_env -c conda-forge python=3.11 suitesparse scikit-sparse scipy numpy pykdtree
```
#### Step 2: Activate the conda environment
```
conda activate clothilde_env
```
#### Step 3: Install the rest with pip
```
pip install polyscope line_profiler
```
---

## 2. High-Level Design

The simulator follows a **state–update paradigm** tailored for control:

* The cloth is represented as a discrete mesh (a nodes matrix `X` + quad connectivity `T`)
* Dynamics are advanced a time step `dt` using a custom implicit time integrator
* External actions, i.e. _grasping and control_ enter as boundary conditions, that is (pin) equality constraints
* Inextensibility and contacts are modeled as hard equality and inequality constraints respectively

---

## 3. Basic Usage

For further details, see the demo and examples folder.

### 3.1 Minimal Example

```python
from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh

# create an initial mesh
na = 20; nb = 33
X, T = createRectangularMesh(a = 0.5, b = 0.8, na = na, nb = nb, h = 0.2)

# call Cloth class
clothilde = Cloth(X, T);

#set default parameters
dt = clothilde.estimateTimeStep(L=0.8)
clothilde.setSimulatorParameters(dt = dt)

#simulate 6 seconds fixing two corners
tf = int(6/dt); inds = [0, na-1]; u = clothilde.positions[inds]
for _ in range(tf):
    clothilde.simulate(u = u, control = inds)

#make a movie with the simulated frames
clothilde.makeMovie(speed = 5, repeat = True, smooth = 2)
```

### 3.2 State Access

The simulator exposes:

* Node positions `self.positions` updated every time `self.simulate()` is called and their history in `self.history_pos`
* Velocities `self.velocities` updated every time `self.simulate()` is called and their history in `self.history_vel`  

---
## 4. Core Inputs

### 4.1 Mesh and Topology (only quad meshes are allowed)

Key mesh parameters:

* initial cloth position `X`: n x 3 positions in Euclidean space of the nodes of the mesh.
* connectivity `T`: quad conectivity of the mesh w.r.t. `X`. Can have non-trivial topology.


### 4.2 Material Parameters

Physical parameters:

* rho:  Surface density (kg/m²)
* delta: Aerodynamics parameter (between 0 and rho, see the first reference)
* alpha: Rayleigh linear damping of big oscillations (usually between rho and 3*rho)
* kappa: stifness or bending resistance (scales like dt²)
* shr: allowed shearing resistance (scales like dt², 0 is no shearing)
* str: allowed stretching resistance (scales like dt², 0 is no stretching)
* mu_f: friction with the floor (typically between 0 and 1)
* mu_s: friction with the cloth itself (typically between 0 and 1)
* thck: adimensional thickness of the cloth (between 0.9 and 1.2)
  
See `self.setSimulatorParameters()` for typical values for a 1m x 1m cloth.   

---

## 6. Time Integration and Solver

### 6.1 Integrator

* Implicit Euler or trapezoidal rule (the second is the default)
  
Time step `dt` is a **critical parameter**:

* Too large → excessive number of iterations
* Too small → unnecessary computational cost

Typical values are of the order of 0.001 and bigger. Use `self.estimateTimeStep(L)` where L is the largest linear dimension of your cloth. 

### 6.2 Constraint satisfaction

We use a custom solver based on XPBD ideas. The critical parameter is `tol` which control relative error in constraint satisfaction to stop iterations. Typical good values are under 1%, e.g. `tol = 0.0075`.


---

## 7. Boundary Conditions and Control Interfaces

The simulator supports:

* Fixed nodes
* Prescribed trajectories 

This allows modeling pick-and-place operations. For this simply call `self.simulate(u, control)` where `u` are the desired m x 3 future positions of the m controled nodes whose indices with respect to `X` are given in the list `control`.  

---

## 8. Citation

If you use this simulator in academic work, please cite:

> An inextensible model for the robotic manipulation of textiles
F Coltraro, J Amorós, M Alberich-Carraminana, C Torras
Applied Mathematical Modelling 101 (2022), 832-858

> A novel collision model for inextensible textiles and its experimental validation
F Coltraro, J Amorós, M Alberich-Carramiñana, C Torras
Applied Mathematical Modelling 128 (2024), 287-3084

---

## 9. Future Extensions

Potential directions:

* Seams
* GPU implementation
* Complex grasps
* Collisions with a moving obstacle (e.g. a gripper)

Contributions and discussions are welcome.


