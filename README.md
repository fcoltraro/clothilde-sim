# clothilde-sim: accurate physical simulation of (quasi)-inextensible textiles 


https://github.com/user-attachments/assets/00d62006-e6ad-4f0a-8ace-b711f61eee81


This repository contains a cloth simulator specifically designed for **robotics and control** tasks. The simulator prioritizes:

* _Inelastic limit_: the simulator is able to simulate (quasi)-inextensible fabrics efficiently.
* _Aerodynamic effects_: interaction with the air can be incorporated (even in the absence of wind).
* _Stability and robustness_: no jittering and stable contact and friction behaviour. 
* _Physical consistency_: the simulator is very stable under various remeshings of the cloth.
* _Ease of use and modularity_: just a few lines of Python code are enough to start simulating.

The simulator has been used and validated in the context of **dynamic textile manipulation** by robots, showcasing its realism when compared with real recordings of various textiles. For more information on robotic applications and validations, click <a href="https://fcoltraro.github.io/projects/telas/"> here</a>.


https://github.com/user-attachments/assets/4dcf4124-d486-414f-9cc4-a79dca4d09b4

---

## 1. Installation

We recommend using Miniconda and a conda environment for simulation (especially in Mac and Windows systems). 

### 1.1 Requirements

The simulator is natively implemented in Python (>=3.11) and relies on the following computing libraries: Numpy, Scipy, scikit-sparse, CHOLMOD and pykdtree. Visualization is done by Polyscope and profiling by line_profiler.

**IMPORTANT**: use scikit-sparse 0.4.x (`>=0.4,<0.5`). Version 0.5.0 changed the syntax of the CHOLMOD Python interface used by the simulator.

### 1.2 Installation Steps

#### Step 0: Clone the repository 
```
git clone https://github.com/fcoltraro/clothilde-sim.git
cd clothilde-sim
```
#### Step 1: Create a new conda environment with the required packages
```
conda create -n clothilde_env -c conda-forge python=3.11 suitesparse "scikit-sparse>=0.4,<0.5" scipy numpy pykdtree
```
#### Step 2: Activate the conda environment
```
conda activate clothilde_env
```
#### Step 3: Install the rest with pip
```
pip install polyscope line_profiler trimesh
```
`trimesh` is used by the gripper examples. Run the minimal example below from `python_code/` so that `implementation` is importable.

---

## 2. High-Level Design

The simulator follows a **state–update paradigm** tailored for control:

* The cloth is represented as a discrete mesh (a nodes matrix `X` + quad connectivity `T`)
* Dynamics are advanced a time step `dt` using a custom implicit time integrator
* External actions, i.e. _grasping and control_ enter as boundary conditions, that is (pin) equality constraints
* Inextensibility and contacts are modeled as hard equality and inequality constraints respectively

---

## 3. Basic Usage

For further details, see the demo and examples folder. Call `setSimulatorParameters()` before starting the simulation.

### 3.1 Minimal Example

```python
from implementation.Cloth import Cloth
from implementation.utils import createRectangularMesh

# create an initial mesh
na = 20; nb = 33
X, T = createRectangularMesh(a = 0.5, b = 0.8, na = na, nb = nb, h = 0.2)
X[:,2] += 0.4 #initial height above the floor

# call Cloth class
clothilde = Cloth(X, T);

#set default parameters
dt = 1/60 #simulate at 60 Hz
clothilde.setSimulatorParameters(dt = dt)

#simulate 6 seconds fixing two corners
tf = int(6/dt); inds = [0, na-1]; u = clothilde.positions[inds]
for _ in range(tf):
    clothilde.simulate(u = u, control = inds)

#make a movie with the simulated frames
clothilde.makeMovie(speed = 1, repeat = True, smooth = 2)
```

### 3.2 State Access

The simulator exposes:

* Node positions `self.positions` updated every time `self.simulate()` is called and their history in `self.history_pos`
* Velocities `self.velocities` updated every time `self.simulate()` is called and their history in `self.history_vel`  

---
## 4. Core Inputs

### 4.1 Mesh and Topology (only quad meshes are allowed)

Key mesh parameters:

* initial cloth position `X`: n x 3 positions in Euclidean space of the nodes of the mesh (metres, z upward).
* connectivity `T`: quad connectivity of the mesh w.r.t. `X`, using zero-based integer indices ordered around each quad. Can have non-trivial topology.


### 4.2 Material Parameters

Physical parameters (defaults in parentheses):

* `rho` (`0.1`): surface density (kg/m²).
* `delta` (`0.1`): aerodynamics / effective gravitational mass parameter (kg/m², between 0 and `rho`, see the first reference). Inertia uses `rho` and gravity uses `delta`; set `delta = rho` for ordinary gravity.
* `alpha` (`0.2`): Rayleigh linear damping of big oscillations (usually between `rho` and `3*rho`).
* `kappa` (`5e-5`): interior bending resistance; larger values give stronger resistance.
* `kappa_bnd` (`5e-6`): bending resistance at the boundary.
* `shr` (`1e-3`): shear compliance; larger values allow more shearing, 0 is the no-shear limit.
* `str` (`1e-6`): stretch compliance; larger values allow more stretching, 0 is the inextensible limit.
* `slf` (`1e-4`): self-contact compliance; larger values give softer contact, 0 is the hard-contact limit.
* `mu_f` (`0.2`): friction with the floor (typically between 0 and 1).
* `mu_s` (`0.35`): friction with the cloth itself (typically between 0 and 1).
* `thck` (`0.95`): dimensionless collision thickness (typically between 0.9 and 1.2). The collision sphere radius is `thck * mean_edge_length / 2.05`.

The solver internally divides `shr`, `str` and `slf` by the squared internal time step; pass the values directly. `kappa` and `kappa_bnd` are used without this rescaling. Gravity is `9.8 m/s²`; stiffness-proportional damping is set internally to `beta = 0.02*kappa`.

See `self.setSimulatorParameters()` for all defaults. Set parameters through this method so that the simulation matrices are updated.

---

## 5. Seams and Environment

To join mesh patches, pass pairs of vertex indices to the constructor: `Cloth(X, T, seams=[[i, j], ...])`. Each pair is constrained to coincide.

The floor is at `z = 0`. To add a static rectangular table, use:

```python
clothilde.addTable(center=[0, 0, 0.4], dimensions=[0.6, 0.5, 0.04], mu=0.3)
```

`center` is its position, `dimensions` its full x/y/z dimensions (metres), and `mu` its friction coefficient.

---

## 6. Time Integration and Solver

### 6.1 Integrator

* Trapezoidal position prediction is the default, followed by constraint corrections and a velocity update from the position difference. For implicit Euler, set `self.implicitEuler = True` after `setSimulatorParameters()`.
  
Time step `dt` is a **critical parameter**:

* Too large → excessive number of iterations
* Too small → unnecessary computational cost

Numerical parameters in `setSimulatorParameters()`:

* `dt` (`1/60` s): time advanced by each call to `simulate()`.
* `sub_steps` (`10`): number of intermediate steps per frame; the internal time step is `dt/sub_steps`, stored in `self.dt`.
* `max_mov` (`0.1`): maximum displacement of controlled nodes per internal step, as a fraction of the mean edge length.

Use positive `dt`, a positive integer `sub_steps`, and positive `max_mov`. Typical internal time steps are of the order of 0.001 s. For example, `dt = 1/60` with `sub_steps = 8` gives an internal time step of `1/(60*8)` s. 

### 6.2 Constraint satisfaction

We use a custom solver based on XPBD ideas. The parameter `tol` (default `0.0075`) controls the normalized constraint residual used to stop iterations. Typical good values are under 1%, e.g. `tol = 0.0075`. The outer solver allows up to 100 iterations per internal step; the self-contact solve uses up to 50 iterations.


---

## 7. Boundary Conditions and Control Interfaces

The simulator supports:

* Fixed nodes
* Prescribed trajectories 

This allows modeling pick-and-place operations. For performing one time-step simply call `self.simulate(u, control)` where `u` are the desired m x 3 future positions of the m controlled nodes whose indices with respect to `X` are given in the list `control`. Targets are interpolated across substeps and limited by `max_mov`. To simulate without controlled nodes, use `self.simulate(u = self.positions[[]], control = [])`.  

---

## 8. Gripper Examples

Two example scripts are provided for interacting with the cloth using a gripper:

* `test_gripper.py`: allows **manual grasping and manipulation** of the cloth using the gripper interface.
* `test_gripper_video.py`: allows **automated grasping and manipulation** by prescribing the gripper trajectory. The gripper `pose` and `jaw_status` can be modified over time to define the desired motion and grasping sequence.

The interface `SimulateGripper(cloth, box_size)` uses a grasp box with full local x/y/z dimensions in metres (default `[0.03, 0.06, 0.02]`). Set its pose with `set_pose(q, p)`, where `q` is a quaternion `[w, x, y, z]` and `p` is its position. Close or release with `set_open(is_open, smooth, center_local=np.zeros(3))`, then call `step()` to advance the cloth.

For the CAD visualization, run the demo with `gripper_cad_files/base.stl`, `jaw_left.stl` and `jaw_right.stl` available at the relative paths used by the script.

https://github.com/user-attachments/assets/29867569-27c6-4484-8a98-d5c4a171192f


## 9. Citation

If you use this simulator in academic work, please cite:

1. **An inextensible model for the robotic manipulation of textiles**  
   F. Coltraro, J. Amorós, M. Alberich-Carramiñana, C. Torras  
   *Applied Mathematical Modelling*, **101** (2022), 832–858

2. **A novel collision model for inextensible textiles and its experimental validation**  
   F. Coltraro, J. Amorós, M. Alberich-Carramiñana, C. Torras  
   *Applied Mathematical Modelling*, **128** (2024), 287–308

3. **[A Simple Gripper Interface for Simulator-Agnostic Cloth Manipulation](https://arxiv.org/abs/2609.29340)**  
   A. Nayak, F. Coltraro, M. Alberich-Carramiñana, C. Torras  
   *arXiv preprint arXiv:2609.29340*, 2026

---

## 10. Future Extensions

Potential directions:

* GPU implementation
* Complex grasps
* Collisions with a moving obstacle (e.g. a gripper)

Contributions and discussions are welcome.


