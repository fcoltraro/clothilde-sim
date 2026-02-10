# Clothilde simulator: accurate physical simulation of textiles 

This repository contains a cloth simulator specifically designed for **robotics and control** tasks. The simulator prioritizes:

* _Inelastic limit_: the simulator is able to simulate (quasi)-inextensible fabrics efficiently.
* _Aerodynamic effects_: interaction with the air can be incorporated (even in the absence of wind).
* _Stability and robustness_: no jittering and stable contact and friction behaviour. 
* _Physical consistency_: the simulator is very stable under various remeshings of the cloth.
* _Easy of use and modularity_: just a few lines of Python code are enough to start simulating.

The simulator has been used and validated in the context of **dynamic textile manipulation** by robots, showcasing its realism when compared with real recordings of various textiles.

---

## 1. Installation

We recomend using the conda package and a conda enviroment for simulation (especially in Mac and Windows systems). 

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

* The cloth is represented as a discrete mesh (nodes + connectivity)
* Dynamics are advanced using an implicit time integrator
* External actions (robot grasps, motions, constraints) enter as boundary conditions or forces

At each simulation step:

1. Boundary conditions and control inputs are applied
2. Internal forces (elastic, damping, aerodynamic) are computed
3. A constrained system is solved to obtain the next state

The design explicitly avoids black-box solvers in favor of transparent and differentiable components.

---

## 3. Basic Usage

### 3.1 Minimal Example

```python
from cloth_simulator import Cloth, Simulator

cloth = Cloth.from_grid(
    nx=20,
    ny=20,
    dx=0.02,
    density=0.2
)

sim = Simulator(
    cloth=cloth,
    dt=0.01
)

for _ in range(200):
    sim.step()
```

This runs a free-fall simulation of a square cloth.

### 3.2 State Access

The simulator exposes:

* Node positions `x`
* Velocities `v`
* Forces (internal and external)

This is intentional: control and estimation algorithms are expected to operate directly on these quantities.

---

## 4. Core Inputs

### 4.1 Mesh and Topology

Key mesh parameters:

* `nx, ny`: number of nodes in each direction
* `dx`: rest edge length
* Connectivity: structural, shear, and bending links

**Recommendation:**

* Use coarse meshes (10–30 nodes per side) for control and MPC
* Finer meshes are possible but solver cost scales quickly

### 4.2 Material Parameters

Typical physical parameters:

* Surface density (kg/m²)
* Stretching stiffness
* Bending stiffness
* Damping coefficients

These parameters are **lumped** and intended to be tuned at the system level rather than matched to textile microstructure.

---

## 5. Aerodynamic Model (If Enabled)

The simulator supports a lightweight aerodynamic correction designed for real-time control.

Characteristics:

* Drag force proportional to relative velocity
* Acts normal to local cloth surface
* Computationally cheap (no CFD, no airflow state)

Key parameters:

* Drag coefficient
* Effective area scaling

**Practical note:**
Aerodynamics significantly improve realism during fast motions but can destabilize the solver if stiffness and damping are not balanced.

---

## 6. Time Integration and Solver

### 6.1 Integrator

* Implicit Euler / semi-implicit scheme
* Chosen for stability under stiff elastic forces

Time step `dt` is a **critical parameter**:

* Too large → excessive numerical damping or solver failure
* Too small → unnecessary computational cost

Typical values:

* `dt = 1e-2` for quasi-static motions
* `dt = 1e-3 – 5e-3` for dynamic manipulation

### 6.2 Solver

The core linear systems are solved using sparse factorizations and OSQP where constraints are active.

Important solver-related parameters:

* Constraint penalty weights
* Solver tolerances
* Maximum iterations

For control applications, solver determinism is prioritized over raw accuracy.

---

## 7. Boundary Conditions and Control Interfaces

The simulator supports:

* Fixed nodes
* Prescribed trajectories (position or velocity controlled)
* Time-varying constraints

This allows modeling:

* Robot grippers
* Sliding contacts
* Pick-and-place operations

Boundary conditions are applied **before** force assembly to ensure consistency.

---

## 8. Parameter Tuning Guidelines

### 8.1 Common Failure Modes

* Excessive stiffness → solver divergence
* Low damping → oscillations
* Large dt + aerodynamics → instability

### 8.2 Recommended Workflow

1. Disable aerodynamics
2. Tune elastic parameters for stability
3. Add damping until oscillations are controlled
4. Enable aerodynamics and reduce `dt` if needed

Always tune parameters **in combination**, not independently.

---

## 9. Performance Considerations

* Computational cost scales roughly linearly with number of nodes
* Cholesky factorizations dominate runtime
* Mesh resolution is the primary performance lever

For MPC or real-time control:

* Prefer coarse meshes
* Use short horizons
* Reuse factorizations where possible

---

## 10. Limitations

This simulator is **not** intended for:

* High-fidelity garment simulation
* Wrinkle-level visual realism
* Self-collision-heavy scenarios

Its purpose is **predictive modeling for control and planning**.

---

## 11. Citation

If you use this simulator in academic work, please cite:

> A Practical Aerodynamic Model for Dynamic Textile Manipulation in Robotics

---

## 12. Future Extensions

Potential directions:

* Differentiable solvers
* Contact with rigid bodies
* Reduced-order cloth models
* Learning-assisted parameter identification

Contributions and discussions are welcome.
