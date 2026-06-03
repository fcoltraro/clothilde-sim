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

EXPORT_DIR = CLOTHILDE_ROOT + "/exported_data3"
sys.path.insert(0, str(CLOTHILDE_ROOT))

from implementation.Cloth_speed import Cloth
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
show_graspbox = False
grasp_box = 0.001 * np.array([40, 40, 40], dtype=float)
tip_center_local = 0.001 * np.array([0.0, 0.0, - (52 - grasp_box[2] * 1000 / 2)], dtype=float)
# tip_center_local = 0.001 * np.array([0.0, 0.0, 0.0], dtype=float)

# Load gripper CAD
def load_mesh(path):
    m = trimesh.load_mesh(path)
    return np.asarray(m.vertices), np.asarray(m.faces)

def transform_mesh(V_local, q, p, local_offset=np.zeros(3)):
    V = np.asarray(V_local, dtype=float) + np.asarray(local_offset, dtype=float).reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V)

V_base0, F_base = load_mesh("gripper_cad_files/base.stl")
V_left0, F_left = load_mesh("gripper_cad_files/jaw_left.stl")
V_right0, F_right = load_mesh("gripper_cad_files/jaw_right.stl")

mesh_scale = 0.001   # change to 0.001 if CAD comes in mm
V_base0 *= mesh_scale
V_left0 *= mesh_scale
V_right0 *= mesh_scale

jaw_open = True
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)
# q0 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0)
# p0 = np.array([0, 0, 0.5])
YELLOW = [186/256, 142/256, 35/256]

# video
repeat = False
fps = 60
dt = 1.0 / fps

state = {
    "k": 0,
    "last_time": time.time()
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

# helper functions to read data
def read_initial_mesh(export_dir):
    vertices_path = os.path.join(export_dir, "mesh_vertices.csv")
    faces_path = os.path.join(export_dir, "mesh_faces.csv")

    vdf = pd.read_csv(vertices_path).sort_values("node_index")
    fdf = pd.read_csv(faces_path).sort_values("face_id")

    X = vdf[["x", "y", "z"]].to_numpy(dtype=float)
    T = fdf[["n0", "n1", "n2", "n3"]].to_numpy(dtype=int)

    return X, T

def read_simulator_parameters(export_dir):
    parameters_path = os.path.join(export_dir, "simulator_parameters.csv")

    pf = pd.read_csv(parameters_path)
    params = pf.iloc[0].to_dict()
    params["sub_steps"] = int(params["sub_steps"])
    
    return params
    
    

def read_cloth_frames(export_dir):
    path = os.path.join(export_dir, "cloth_frames.csv")

    df = pd.read_csv(path)
    df = df.sort_values(["frame", "node_index"])

    frames = []

    for frame_id, group in df.groupby("frame", sort=True):
        X = group[["x", "y", "z"]].to_numpy(dtype=float)
        frames.append(X)

    return frames

def read_gripper_poses(export_dir):
    path = os.path.join(export_dir, "gripper_poses.csv")

    df = pd.read_csv(path)
    df = df.sort_values("frame")

    p_gripper = []
    q_gripper = []
    jaw_status = []

    for frame_id, group in df.groupby("frame", sort=True):
        row = group.iloc[0]
        p = row[["px", "py", "pz"]].to_numpy(dtype=float)
        q = row[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
        # j = row[["jaw_open"]].to_numpy(dtype=int)
        j = int(row["jaw_open"])
        p = p - tip_center_local
        p_gripper.append(p)
        q_gripper.append(q)
        jaw_status.append(j)

    return p_gripper, q_gripper, jaw_status

X0, T = read_initial_mesh(EXPORT_DIR)
frames = read_cloth_frames(EXPORT_DIR)
n_frames = len(frames)

print("Mesh:", X0.shape)
print("Faces:", T.shape)
print("Frames:", len(frames))

X_reset = X0.copy()

params = read_simulator_parameters(EXPORT_DIR)

cloth = Cloth(X0, T) # already gets the width, height, na, nb

cloth.setSimulatorParameters(**params)
cloth.preparePolyscope()    
cloth.rad = 0.005

cloth2 = Cloth(frames[0].copy(), T)
cloth2.rad = 0.005
cloth2.label = "recorded_cloth"

phi2_0 = cloth2.Am @ cloth2.positions 

# for _ in range(2):
#     phi2_0 = cloth2.S @ phi2_0

rec_mesh = ps.register_surface_mesh(
    cloth2.label,
    phi2_0,
    cloth2.triangles,
    color=[0.55, 0.90, 0.55],
    transparency=0.45,
    smooth_shade=True,
    edge_width=0
)

rec_pc = ps.register_point_cloud(
    cloth2.label + "_nodes",
    cloth2.positions,
    enabled=False
)

# if show_gripper:
p_gripper, q_gripper, jaw_status = read_gripper_poses(EXPORT_DIR)

grip = SimulateGripper(cloth, box_size=grasp_box)

base_mesh = ps.register_surface_mesh(
    "gripper_base",
    transform_mesh(V_base0, q_gripper[0], p_gripper[0]),
    F_base, color=YELLOW
)
left_mesh = ps.register_surface_mesh(
    "gripper_left",
    transform_mesh(V_left0, q_gripper[0], p_gripper[0], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
right_mesh = ps.register_surface_mesh(
    "gripper_right",
    transform_mesh(V_right0, q_gripper[0], p_gripper[0], local_offset=np.array([ jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)

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
    
    # cloth2.positions = frames[k]
    # ps.get_surface_mesh(cloth2.label).update_vertex_positions(cloth2.positions)
    
    cloth2.positions = frames[k]

    phi2_all = cloth2.Am @ cloth2.positions

    # for _ in range(2):
    #     phi2_all = cloth2.S @ phi2_all

    rec_mesh.update_vertex_positions(phi2_all)
    rec_pc.update_point_positions(cloth2.positions)
    
    p = grip.p.copy()
    q = grip.q.copy()
    
    current_gap = jaw_gap_open if jaw_open else jaw_gap_closed

    base_mesh.update_vertex_positions(
        transform_mesh(V_base0, q, p)
    )

    left_mesh.update_vertex_positions(
        transform_mesh(
            V_left0,
            q,
            p,
            local_offset=np.array([-current_gap / 2, 0, 0])
        )
    )

    right_mesh.update_vertex_positions(
        transform_mesh(
            V_right0,
            q,
            p,
            local_offset=np.array([current_gap / 2, 0, 0])
        )
    )
    
def callback():
    global jaw_open, grasp_box, tip_center_local, n_frames
    global p_gripper, q_gripper, jaw_status
    
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
    grip.set_open(is_open=jaw_open, smooth=smooth, box=grasp_box, center_local=tip_center_local)
    grip.step()
    
    update_scene()

    state["k"] += 1

    # reset
    if state["k"] >= n_frames:
        
        if repeat:
            state["k"] = 0 # repeat
        
            cloth.positions = X_reset.copy()
            cloth2.positions = X_reset.copy()
        else:
            state["k"] = n_frames - 1
            
ps.set_user_callback(callback)
ps.show() # comment this while debugging. The warning is due to this line
ps.clear_user_callback()

# cloth.makeMovie(speed=1, repeat=True, smooth=2)
    