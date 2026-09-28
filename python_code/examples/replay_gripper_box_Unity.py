# Replay the data exported from Unity (in binary form)

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

EXPORT_DIR = CLOTHILDE_ROOT + "/exported_data3_binary"
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
grasp_box = 0.001 * np.array([40, 40, 40], dtype=float)
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
def read_unity_binary(path):
    with open(path, "rb") as f: # rb: read binary
        magic = f.read(8).decode("ascii")
    
        if magic != "CLTHSIM1":
            raise ValueError(f"Wrong file format: {magic}")
        
        version = np.fromfile(f, dtype="<i4", count=1)[0]
        # <:   little-endian i4: 4-byte integer (can store upto 4 * 8 = 32 bits)
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
        header = np.fromfile(f, dtype="<i4", count=4)
        n_frames, n_vertices, n_faces, n_params = header
        print(f'n_frames: {n_frames}, n_vertices: {n_vertices}, n_faces: {n_faces}, n_params: {n_params}')
        
        mesh_vertices_bin = np.fromfile(f, dtype="<f8", count=n_vertices * 3)
        mesh_vertices = mesh_vertices_bin.reshape(n_vertices, 3)
        
        mesh_faces_bin = np.fromfile(f, dtype="<i4", count=n_faces * 4)
        mesh_faces = mesh_faces_bin.reshape(n_faces, 4)
        
        cloth_frames_bin = np.fromfile(f, dtype="<f8", count=n_frames * n_vertices * 3)
        cloth_frames = cloth_frames_bin.reshape(n_frames, n_vertices, 3)
        
        gripper_poses_bin = np.fromfile(f, dtype="<f4", count=n_frames * 7)
        gripper_poses = gripper_poses_bin.reshape(n_frames, 7)
        
        jaw_status = np.fromfile(f, dtype="<i4", count=n_frames)
        
        times = np.fromfile(f, dtype="<f4", count=n_frames)
        
        sim_params_arr = np.fromfile(f, dtype="<f4", count=n_params)
        
    param_names = [
        "dt", "tol", "sub_steps", "rho", "delta", "alpha",
        "kappa", "kappa_bnd", "str", "shr", "slf",
        "mu_f", "mu_s", "thck"
    ]
    params = dict(zip(param_names, sim_params_arr))
    params["sub_steps"] = int(params["sub_steps"])
    
    return mesh_vertices, mesh_faces, cloth_frames, gripper_poses, jaw_status, times, params

#region define clothes

X0, T0, frames, gripper_poses, jaw_status, times, params = read_unity_binary(EXPORT_DIR + "/traj_mantel_single.bin")
n_frames = len(frames)

print("Mesh:", X0.shape)
print("Faces:", T0.shape)
print("Frames:", len(frames))
print(np.max(np.abs(frames[0] - X0)))

X_reset = X0.copy()

cloth = Cloth(X0.copy(), T0.copy()) # already gets the width, height, na, nb

params["dt"] = dt
cloth.setSimulatorParameters(**params)

cloth.preparePolyscope()

cloth_unity = Cloth(X0.copy(), T0.copy())

cloth_unity.positions = X0.copy()

cloth_unity.label = "recorded_cloth"
phi2_0 = cloth_unity.Am @ cloth_unity.positions 

for _ in range(2):
    phi2_0 = cloth_unity.S @ phi2_0

rec_mesh = ps.register_surface_mesh(
    cloth_unity.label,
    phi2_0,
    cloth_unity.triangles,
    color=[0.55, 0.90, 0.55],
    transparency=0.45,
    smooth_shade=True,
    edge_width=0
)

rec_pc = ps.register_point_cloud(
    cloth_unity.label + "_nodes",
    cloth_unity.positions,
    enabled=False
)

# if show_gripper:
p_gripper = gripper_poses[:, 0:3]
q_gripper = gripper_poses[:, 3:7]

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
    
    # recorded cloth
    k = state["k"]
    
    # cloth_unity.positions = frames[k]
    # ps.get_surface_mesh(cloth_unity.label).update_vertex_positions(cloth_unity.positions)
    
    cloth_unity.positions = frames[k]
    # cloth_unity.velocities = velocities[k]

    phi2_all = cloth_unity.Am @ cloth_unity.positions

    for _ in range(2):
        phi2_all = cloth_unity.S @ phi2_all

    rec_mesh.update_vertex_positions(phi2_all)
    rec_pc.update_point_positions(cloth_unity.positions)
    
    p = grip.p.copy()
    q = grip.q.copy()
    
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
        
        if repeat:
            state["k"] = 0 # repeat
        
            cloth.positions = X_reset.copy()
            cloth_unity.positions = X_reset.copy()
        else:
            state["k"] = n_frames - 1
        return
            
ps.set_user_callback(callback)
ps.show() # comment this while debugging. The warning is due to this line
ps.clear_user_callback()

# cloth.makeMovie(speed=1, repeat=True, smooth=2)