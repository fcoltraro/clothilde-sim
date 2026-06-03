import sys, os
import numpy as np
import pandas as pd
import trimesh
import polyscope as ps

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))
sys.path.append(parent_dir + "/python_code")
# export_dir = "Z:\IRI_2026\clothilde-sim\python_code\exported_data3"

CLOTHILDE_ROOT = parent_dir + "/python_code"
sys.path.append(CLOTHILDE_ROOT)

EXPORT_DIR = CLOTHILDE_ROOT + "/exported_data3"
sys.path.insert(0, str(CLOTHILDE_ROOT))

from implementation.Cloth import Cloth
from implementation.Gripper import (
    SimulateGripper, 
    quat_transform_points,
    quat_normalize,
    quat_from_axis_angle
)


# Gripper data

show_gripper = True

def load_mesh(path):
    m = trimesh.load_mesh(path)
    return np.asarray(m.vertices), np.asarray(m.faces)

def transform_mesh(V_local, q, p, local_offset=np.zeros(3)):
    V = np.asarray(V_local, dtype=float) + np.asarray(local_offset, dtype=float).reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V)

# load 3 parts
V_base0, F_base = load_mesh("gripper_cad_files/base.stl")
V_left0, F_left = load_mesh("gripper_cad_files/jaw_left.stl")
V_right0, F_right = load_mesh("gripper_cad_files/jaw_right.stl")

mesh_scale = 0.001   # change to 0.001 if CAD comes in mm
V_base0 *= mesh_scale
V_left0 *= mesh_scale
V_right0 *= mesh_scale

# if cloth.polyscoped is False:
#     cloth.preparePolyscope()
jaw_open = True
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)
# q0 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0)
# p0 = np.array([0, 0, 0.5])
YELLOW = [186/256, 142/256, 35/256]


# helper functions to read data

def read_initial_mesh(export_dir):
    vertices_path = os.path.join(export_dir, "mesh_vertices.csv")
    faces_path = os.path.join(export_dir, "mesh_faces.csv")

    vdf = pd.read_csv(vertices_path).sort_values("node_index")
    fdf = pd.read_csv(faces_path).sort_values("face_id")

    X = vdf[["x", "y", "z"]].to_numpy(dtype=float)
    T = fdf[["n0", "n1", "n2", "n3"]].to_numpy(dtype=int)

    return X, T


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

    for frame_id, group in df.groupby("frame", sort=True):
        row = group.iloc[0]
        p = row[["px", "py", "pz"]].to_numpy(dtype=float)
        q = row[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
        p_gripper.append(p)
        q_gripper.append(q)

    return p_gripper, q_gripper

def slow_down_frames(frames, factor=2):
    slow_frames = []
    
    for i in range(len(frames) - 1):
        X0 = frames[i]
        X1 = frames[i+1]
        
        for k in range(factor):
            alpha = k / factor
            X = X0 * (1 - alpha) + X1 * alpha
            slow_frames.append(X)
        
    slow_frames.append(frames[-1])
            
    return slow_frames

def main():
    X0, T = read_initial_mesh(EXPORT_DIR)
    frames = read_cloth_frames(EXPORT_DIR)
    
    slowed_frames = slow_down_frames(frames, 2)

    cloth = Cloth(X0, T)
    
    cloth.preparePolyscope()    
    cloth.rad = 0.005
        
    # # if show_gripper:
    # p_gripper, q_gripper = read_gripper_poses(EXPORT_DIR)
    
    # grip = SimulateGripper(cloth, box_size=np.array([0.06, 0.01, 0.015], dtype=float))
    
    # base_mesh = ps.register_surface_mesh(
    #     "gripper_base",
    #     transform_mesh(V_base0, q_gripper[0], p_gripper[0]),
    #     F_base, color=YELLOW
    # )
    # left_mesh = ps.register_surface_mesh(
    #     "gripper_left",
    #     transform_mesh(V_left0, q_gripper[0], p_gripper[0], local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    #     F_left, color=YELLOW
    # )
    # right_mesh = ps.register_surface_mesh(
    #     "gripper_right",
    #     transform_mesh(V_right0, q_gripper[0], p_gripper[0], local_offset=np.array([ jaw_gap_open/2, 0, 0])),
    #     F_right, color=YELLOW
    # )
    
    # smooth = 2
    # n_frames = len(frames)
    # for k in range(n_frames):
        
    #     cloth.positions = frames[k]
        
    #     phi_mat = frames[k]

    #     # Convert physical nodes to displayed/rendered surface
    #     phi_all = cloth.Am @ phi_mat

    #     # Smooth displayed surface
    #     for _ in range(smooth):
    #         phi_all = cloth.S @ phi_all

    #     # Update displayed surface mesh
    #     ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    #     # Update raw node point cloud
    #     ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)
        
    #     # Update gripper pose
    #     grip.set_pose(q_gripper[k], p_gripper[k])

    #     # Update gripper CAD meshes
    #     base_mesh.update_vertex_positions(
    #         transform_mesh(V_base0, q_gripper[k], p_gripper[k])
    #     )

    #     left_mesh.update_vertex_positions(
    #         transform_mesh(
    #             V_left0,
    #             q_gripper[k],
    #             p_gripper[k],
    #             local_offset=np.array([-jaw_gap_open / 2, 0, 0])
    #         )
    #     )

    #     right_mesh.update_vertex_positions(
    #         transform_mesh(
    #             V_right0,
    #             q_gripper[k],
    #             p_gripper[k],
    #             local_offset=np.array([jaw_gap_open / 2, 0, 0])
    #         )
    #     )
        # # This tells Polyscope to draw the next frame
        # ps.frame_tick()
    
    # else:
    cloth.history_pos = frames
    cloth.positions = frames[-1]
     
    print("Mesh:", X0.shape)
    print("Faces:", T.shape)
    print("Frames:", len(frames))

    cloth.makeMovie(speed=1, repeat=True, smooth=2)

if __name__ == "__main__":
    main()