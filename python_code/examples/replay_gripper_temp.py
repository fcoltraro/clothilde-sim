# Replay a simulation from the exported files of Python (in .npz)
# to compare with another Python simulation

import sys, os
import numpy as np
import pandas as pd
import trimesh
import polyscope as ps
import time

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))
sys.path.append(parent_dir + "/python_code")
# export_dir = "Z:\IRI_2026\clothilde-sim\python_code\exported_data3"

CLOTHILDE_ROOT = parent_dir + "/python_code"
sys.path.append(CLOTHILDE_ROOT)

# EXPORT_DIR = CLOTHILDE_ROOT + "/exported_data3_box_old"
# EXPORT_DIR = CLOTHILDE_ROOT + "/exported_test_gripper2" # By changing some float64 values to
# # float32 in Cloth.py
EXPORT_DIR = CLOTHILDE_ROOT + "/exported_test_gripper_temp" # By keeping everything float64
sys.path.insert(0, str(CLOTHILDE_ROOT))

from implementation.Cloth_speed import Cloth
from implementation.utils import createRectangularMesh
from implementation.Gripper import (
    SimulateGripper, 
    quat_transform_points,
    quat_normalize,
    quat_from_axis_angle
)

smooth = 2
# Gripper data
show_gripper = True

# grasp box
show_graspbox = True
grasp_box = 0.001 * np.array([30, 30, 30], dtype=float)
# grasp_box = 0.001 * np.array([40, 40, 40], dtype=float)
# tip_center_local = 0.001 * np.array([0.0, 0.0, - (52 - grasp_box[2] * 1000 / 2)], dtype=float)
tip_center_local = 0.001 * np.array([0.0, 0.0, 0.0], dtype=float)

# video
repeat = False
fps = 60
dt = 1 / fps

state = {
    "k": 0,
    "last_time": time.time(),
    "paused": False
}

# graspbox geometry
if show_graspbox:
    box_faces = np.array([
        [0,1,2], [0,2,3],
        [4,5,6], [4,6,7],
        [0,1,5], [0,5,4],
        [1,2,6], [1,6,5],
        [2,3,7], [2,7,6],
        [3,0,4], [3,4,7],
    ], dtype=int)

    def get_box_vertices_world_offset(p, q, box_size, center_local):
        hx, hy, hz = 0.5 * np.asarray(box_size, dtype=float)
        c = np.asarray(center_local, dtype=float).reshape(3,)

        V_local = np.array([
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ], dtype=float)

        V_local = V_local + c.reshape(1, 3)
        return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V_local)

#region read data
# def read_initial_mesh(export_dir):
#     initial_state = np.load(os.path.join(export_dir, "initial_state.npz"))
#     X = initial_state["X"]
#     T = initial_state["T"]
#     return X, T

# def read_simulator_parameters(export_dir):
#     parameters_path = os.path.join(export_dir, "simulator_parameters.csv")

#     pf = pd.read_csv(parameters_path)
#     params = pf.iloc[0].to_dict()
#     params["sub_steps"] = int(params["sub_steps"])
#     params["dt"] = dt
    
#     return params

# def read_cloth_frames(export_dir):
#     cloth_frames = np.load(os.path.join(export_dir, "cloth_frames.npz"))
#     frames = cloth_frames["frames"]
#     times = cloth_frames["t"]
#     return frames

def read_gripper_poses(export_dir):
    gripper_poses = np.load(os.path.join(export_dir, "traj_mantel.npz"))
    times = gripper_poses["t"]
    p_gripper = gripper_poses["position"]
    q_gripper = gripper_poses["quaternion"]
    jaw_status = gripper_poses["jaw_status"].astype(int)
    return p_gripper, q_gripper, jaw_status, times

#region define clothes

# X0_f32 = X0.astype(np.float32).astype(np.float64)
# frames_f32 = frames.astype(np.float32).astype(np.float64)
# X0 = X0_f32.copy()
# frames = frames_f32.copy()

na = 23
nb = 23

side_a = 0.5
side_b = 0.7

X, T = createRectangularMesh(a=side_a, b=side_b, na=na, nb=nb, h=0.0)

cloth = Cloth(X, T)
dt = cloth.estimateTimeStep(L=0.8)
# cloth.setSimulatorParameters(dt=dt, shr=0.1 * 0.0001, str=0.001 * 0.0001)
cloth.setSimulatorParameters(dt=1/60, sub_steps=6, slf=0.01)

if cloth.polyscoped is False:
    cloth.preparePolyscope()

# if show_gripper:
p_gripper, q_gripper, jaw_status, times = read_gripper_poses(EXPORT_DIR)
# print(f'gripper start position: {p_gripper[180]}')
n_frames = len(p_gripper)

grip = SimulateGripper(cloth, box_size=grasp_box)

if show_graspbox:
    V_dbg = get_box_vertices_world_offset(p=p_gripper[0], q=q_gripper[0], box_size=grasp_box, center_local=tip_center_local)

    ps.register_surface_mesh(
        "grasp_box",
        V_dbg,
        box_faces,
        color=[1.0, 1.0, 0.0],
        transparency=0.55,
        material="wax"
    )
    
#region update scene    
def update_scene():
    # Update cloth visualization
    phi_mat = cloth.positions

    phi_all = cloth.Am @ phi_mat
    for _ in range(2):
        phi_all = cloth.S @ phi_all

    if len(grip.controlled) > 0:
        ctrl = np.asarray(grip.controlled, dtype=int)
        phi_all[ctrl] = phi_mat[ctrl]

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)
    
        
    
#region callback
    
def callback():
    global jaw_open, grasp_box, tip_center_local, n_frames
    global p_gripper, q_gripper, jaw_status
    
    if state["paused"]:
        return
    
    now = time.time()

    if now - state["last_time"] < dt:
        return

    state["last_time"] = now

    k = state["k"]

    q = q_gripper[k]
    p = p_gripper[k]
    jaw_open = bool(jaw_status[k])
        
    if show_graspbox:
        V_dbg = get_box_vertices_world_offset(p, q, grasp_box, tip_center_local)
        ps.get_surface_mesh("grasp_box").update_vertex_positions(V_dbg)

        if jaw_open:
            ps.get_surface_mesh("grasp_box").set_color([0.0, 1.0, 0.0])
        else:
            ps.get_surface_mesh("grasp_box").set_color([1.0, 0.0, 0.0])
    
    grip.set_pose(q_gripper[k], p_gripper[k])
    # nodes detected when open --> close
    grip.set_open(is_open=jaw_open, smooth=smooth, box=grasp_box, 
                  center_local=tip_center_local, squeeze_enabled=True)
    grip.step()
    
    update_scene()

    state["k"] += 1

    # reset
    if state["k"] >= n_frames:
        
        state["k"] = n_frames - 1
        state["paused"] = True
        print("Reached final frame. Simulation stopped.")
            
ps.set_user_callback(callback)
ps.show() # comment this while debugging. The warning is due to this line
ps.clear_user_callback()

# cloth.makeMovie(speed=1, repeat=True, smooth=2)
    