# Gripper chooses the nodes where the grasp box overlaps with the 
# AABB of quadrilateral grasped and selects all four nodes

import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")
CLOTHILDE_ROOT = parent_dir + "/python_code"
EXPORT_DIR = CLOTHILDE_ROOT + "/exported_test_gripper_temp" # By keeping everything float64
# os.makedirs(EXPORT_DIR, exist_ok=True)
export = False

from implementation.Cloth_speed import Cloth
from implementation.utils import createRectangularMesh

is_gripper_box = True # True: box, False: square pyramid

if is_gripper_box:
    from implementation.Gripper import (
        SimulateGripper,
        quat_from_axis_angle,
        quat_to_rotmat,
        quat_transform_points,
        quat_normalize,
        quat_from_rotvec,
        quat_rotate_vector,
        quat_mul
    )
else:
    from implementation.Gripper_pyramid import (
        SimulateGripper,
        quat_from_axis_angle,
        quat_to_rotmat,
        quat_transform_points,
        quat_normalize,
        quat_from_rotvec,
        quat_rotate_vector,
        quat_mul
    )
    
import numpy as np
import trimesh

# na = 24
# nb = 20
na = 23
nb = 28

side_a = 0.42
side_b = 0.59

X, T = createRectangularMesh(a=side_a, b=side_b, na=na, nb=nb, h=0.0)
# X[:, 2] += 0.7
# X += 0.0001 * np.random.randn(X.shape[0], 3)

cloth = Cloth(X, T)
dt = cloth.estimateTimeStep(L=0.8)
cloth.setSimulatorParameters(dt = 1/60, tol = 0.0075, sub_steps = 6,
                               rho = 0.3046, delta = 0.139, alpha = 0.416,
                               kappa = 0.25*1e-4, kappa_bnd = 0, 
                               str = 0.01*1e-4, shr = 5*1e-4, slf = 1*1e-4,
                               mu_f = 0.2, mu_s = 0.60, thck = 0.95, max_mov= 0.1)

grip = SimulateGripper(cloth, box_size=np.array([0.06, 0.01, 0.015], dtype=float))

import polyscope as ps
import polyscope.imgui as psim

smooth=2 # displayed cloth surface may not be the raw vertex positions
# Hence while grasping, the cloth might not sit between the jaws

# initial pose controls 
gripper_pos = cloth.positions.mean(axis=0).copy()
gripper_pos[2] = 0.5

jaw_open = True
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)
# 30 mm - 6 mm (extension length) is the gap between the grippers in my CAD model
YELLOW = [186/256, 142/256, 35/256]

rotvec = np.array([0.0, 0.0, 0.0], dtype=float)

q0 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0)
grip.set_pose(q0, gripper_pos)
q0 = grip.q.copy()
p0 = grip.p.copy()

### helper functions
follow_offset = None
follow_enabled = False

# Gripper parallelopiped

if is_gripper_box:
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


else:
    box_faces = np.array([
        [0,1,4], [1,2,4],
        [2,3,4], [0,3,4],
        [0,1,2], [0,2,3]], dtype=int)

    def get_box_vertices_world_offset(p, q, box_size, center_local):
        side, height = np.asarray(box_size, dtype=float)
        c = np.asarray(center_local, dtype=float).reshape(3,)

        V_local = np.array([
            [-side/2, -side/2, 0],
            [-side/2, side/2, 0],
            [side/2, side/2, 0],
            [side/2, -side/2, 0],
            [0, 0, -height]
        ], dtype=float)

        V_local = V_local + c.reshape(1, 3)
        return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V_local)

def load_mesh(path):
    m = trimesh.load_mesh(path)
    return np.asarray(m.vertices), np.asarray(m.faces)

def transform_mesh(V_local, q, p, local_offset=np.zeros(3)):
    V = np.asarray(V_local, dtype=float) + np.asarray(local_offset, dtype=float).reshape(1, 3)
    return quat_transform_points(np.asarray(p, dtype=float), quat_normalize(q), V)

###
# load 3 parts
V_base0, F_base = load_mesh("gripper_cad_files/base.stl")
V_left0, F_left = load_mesh("gripper_cad_files/jaw_left.stl")
V_right0, F_right = load_mesh("gripper_cad_files/jaw_right.stl")

mesh_scale = 0.001   # change to 0.001 if CAD comes in mm
V_base0 *= mesh_scale
V_left0 *= mesh_scale
V_right0 *= mesh_scale

if cloth.polyscoped is False:
    cloth.preparePolyscope()

ps.register_surface_mesh(
    "gripper_base",
    transform_mesh(V_base0, q0, p0),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left",
    transform_mesh(V_left0, q0, p0, local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right",
    transform_mesh(V_right0, q0, p0, local_offset=np.array([ jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)

# grasp box
if is_gripper_box:
    grasp_box = 0.001 * np.array([30, 30, 30], dtype=float)
    tip_center_local = 0.001 * np.array([0.0, 0.0, - (52 - grasp_box[2] * 1000 / 2)], dtype=float)
else:    
    grasp_box = 0.001 * np.array([30, 15], dtype=float)
    tip_center_local = 0.001 * np.array([0.0, 0.0, - (52 - grasp_box[1] * 1000 / 2)], dtype=float)


# grasp_box = 0.001 * np.array([6, 30, 6], dtype=float)
# tip_center_local = 0.001 * np.array([0.0, 0.0, -49], dtype=float)

# 37 - 3 + 15 mm (check CAD file)  is the distance of the jaw extensions 
# from the origin of the gripper frame

V_dbg = get_box_vertices_world_offset(p=grip.p, q=grip.q, box_size=grasp_box, center_local=tip_center_local)

ps.register_surface_mesh(
    "grasp_box",
    V_dbg,
    box_faces,
    color=[1.0, 1.0, 0.0],
    transparency=0.55,
    material="wax"
)

# initialize grasped nodes

# ps.register_point_cloud("grasped_nodes", np.zeros((0,3)), radius=0.002, color=[1.0, 0.0, 0.0])

#region export
export_frame = 0
export_times = []
cloth_frames = []

def record_frame():
    global export_frame

    t = export_frame * frame_dt
    cloth_frames.append(cloth.positions.copy())
    export_times.append(t)
            
    export_frame += 1

def save_dynamic_data():
    outfile = os.path.join(EXPORT_DIR, "cloth_frames.npz")
    np.savez(outfile,
        frames=np.asarray(cloth_frames, dtype=np.float64),
        t=np.asarray(export_times, dtype=np.float64)
        )

#region update scene
# only for visualization (redraw/update): also called every frame through callback
def update_scene():
    # copied from Cloth.py: to update meshes
    phi_mat = cloth.positions
    phi_all = cloth.Am @ phi_mat
    for _ in range(smooth):
        phi_all = cloth.S @ phi_all

    # force displayed vertices at grasped nodes to match true solver nodes
    # So that the grasped node is in between the jaws
    if len(grip.controlled) > 0:
        ctrl = np.asarray(grip.controlled, dtype=int)
        # phi_all[ctrl] = phi_mat[ctrl]

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)

    q = grip.q.copy()
    p = grip.p.copy()

    current_gap = jaw_gap_open if jaw_open else jaw_gap_closed

    ps.get_surface_mesh("gripper_base").update_vertex_positions(
        transform_mesh(V_base0, q, p)
    )
    ps.get_surface_mesh("gripper_left").update_vertex_positions(
        transform_mesh(V_left0, q, p, local_offset=np.array([-current_gap/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right").update_vertex_positions(
        transform_mesh(V_right0, q, p, local_offset=np.array([ current_gap/2, 0, 0]))
    )

    # update grasp box
    V_dbg = get_box_vertices_world_offset(grip.p, grip.q, grasp_box, tip_center_local)
    ps.get_surface_mesh("grasp_box").update_vertex_positions(V_dbg)

    if jaw_open:
        ps.get_surface_mesh("grasp_box").set_color([0.0, 1.0, 0.0])
    else:
        ps.get_surface_mesh("grasp_box").set_color([1.0, 0.0, 0.0])

    # update gripper frame from current pose
    R = quat_to_rotmat(grip.q)
    try:
        ps.remove_structure("gripper_frame_live")
    except:
        pass

    origin = np.asarray(grip.p + R @ tip_center_local, dtype=float).reshape(1, 3)
    # pc = ps.register_point_cloud("gripper_frame_live", origin, radius=0.005)
    # pc.add_vector_quantity("x", (0.08 * R[:, 0]).reshape(1, 3), vectortype="ambient", enabled=True, color=[1.0, 0.0, 0.0])
    # pc.add_vector_quantity("y", (0.08 * R[:, 1]).reshape(1, 3), vectortype="ambient", enabled=True, color=[0.0, 1.0, 0.0])
    # pc.add_vector_quantity("z", (0.08 * R[:, 2]).reshape(1, 3), vectortype="ambient", enabled=True, color=[0.0, 0.0, 1.0])
    
    # try:
    #     ps.remove_structure("grasped_nodes")
    # except:
    #     pass

    if len(grip.controlled) > 0:
        ps.register_point_cloud("grasped_nodes", phi_all[ctrl], radius=0.002, color=[1.0, 0.0, 0.0])
        # ps.get_point_cloud("grasped_nodes").update_point_positions(phi_all[ctrl])
    else:
        ps.register_point_cloud("grasped_nodes", np.zeros((0,3)), radius=0.002, color=[1.0, 0.0, 0.0])

#region ideal fold

# #crease along the x-axis
# X0_ref = cloth.positions.copy()
# # Nodes on this side are folded.
# # For your corner-fold setup, this usually means y > 0.
# fold_side = X0_ref[:, 1] > 0.0 # Boolean with shape (N,)
# print(f'fold side: {fold_side.shape}')
# Xideal = X0_ref.copy()
# Xideal[fold_side, 1] = -X0_ref[fold_side, 1] # change y to -y

### From a file
sys.path.insert(0, str(CLOTHILDE_ROOT))
cloth_positions = np.load(os.path.join(EXPORT_DIR, "cloth_frames.npz"))
frames = cloth_positions["frames"]
X0_ref = frames[-1]
Xideal = X0_ref.copy()

fold_time = []
fold_error = []

## Plotting

# Xideal[:, 2] += 0.02

# Xideal_all = cloth.Am @ Xideal
# for _ in range(smooth):
#     Xideal_all = cloth.S @ Xideal_all
# ps.register_surface_mesh(
#     "ideal_cloth",
#     Xideal_all,
#     cloth.triangles,
#     color=[1.0, 0.2, 1.0],
#     transparency=0.5,
#     smooth_shade=True,
#     edge_width=1
# )

def fold_error_to_final(X, X_ideal, side_a, side_b):
    """
    Normalized RMS distance from current cloth shape
    to the final ideal folded cloth.
    """
    d_cloth = np.sqrt(side_a**2 + side_b**2)

    err = X - X_ideal

    return np.sqrt(np.mean(np.sum(err**2, axis=1))) / d_cloth

#region folding
# called every frame: interactive frame loop
# step(): one physics step per frame
frame_id = 0
frame_dt = cloth.frame_rate

p_a1 = p0.copy()
q_a1 = q0.copy()

#### Cases
case = 7
# is_rotate_gripper = True

if case == 0:
    # pyramid; no rotation; 1 node: 643
    is_rotate_gripper = False
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [0, 0, 0.01] 
if case == 1:
    # box; no rotation; 2 nodes: 642, 643; 
    is_rotate_gripper = False
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [0, 1/2 * (side_b / nb), 0.01] 
if case == 2:
    # box; no rotation; 2 nodes: 620, 643; 
    is_rotate_gripper = False
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [1/2 * (side_a / na), 0, 0.01]
if case == 3:
    # box; no rotation; 4 nodes: 619, 620, 642, 643; 
    is_rotate_gripper = False
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [0, 0, 0.01] 
    # p_b1 = cloth.positions[len(cloth.positions) - 1] + [- 1/8 * (side_a / na), 0, 0] 

if case == 4:
    # box; no rotation; 2 nodes: 643
    is_rotate_gripper = True
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [1/2 * (side_a / na), 3/2 * (side_b / nb), 0]
if case == 5:
    # box; no rotation; 2 nodes: 642, 643
    is_rotate_gripper = True
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [0, 3/2 * (side_b / nb), 0] 
if case == 6:
    # box; no rotation; 2 nodes: 620, 643
    is_rotate_gripper = True
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [1/2 * (side_a / na), (side_b / nb), 0]
if case == 7:
    # box; no rotation; 4 nodes: 619, 620, 642, 643
    is_rotate_gripper = True
    p_b1 = cloth.positions[len(cloth.positions) - 1] + [0, (side_b / nb), 0]

if is_rotate_gripper:
    q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
                        quat_from_axis_angle([0, 1, 0], -np.pi/4))
else:
    q_b1 = quat_from_axis_angle([0, 0, 1], -np.pi/2) # no rotation
    # q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
    #                     quat_from_axis_angle([0, 1, 0], -np.pi/4))

p_c1 = np.array([side_a/2, -side_b/2, 0.04])
if is_rotate_gripper:
    q_c1 = quat_mul(q_b1.copy(), quat_from_axis_angle([0, 1, 0], np.pi/4))
else:
    q_c1 = q_b1.copy()

## Start directly from a corner
T1 = 1.0   # seconds: stay at initial pose, gripper open
T2 = 1.0   # close gripper
T3 = 3.0   # parabolic motion (follows a quadratic Bezier curve)
T_cam1 = 3.0

# T_end = T1 + T2 + T3 + T4 + T_cam1 + 0.5  # small hold after motion

T_end = T1 + T2 + T3 + 3.0  # small hold after motion

def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    # return s * s * (3.0 - 2.0 * s)
    return 10 * s**3 - 15 * s**4 + 6 * s**5
# standard smoothing function: that gives f'(0) = f'(1) = 0

parabola_locus = []

def planned_pose_1(t):
    # First fold
    global p_d1, p_e1, p_g1, p_i1
    if t < T1:
        return p_b1, q_b1, True
    
    if t < T1 + T2:
        return p_b1, q_b1, False
    
    if t < T1 + T2 + T3:
        tau = (t - T1 - T2) / T3
        s = smoothstep(tau)
        
        pv = np.array([side_a/2, side_b/2, 0.18]) # control handle to pull the parabola
        # ps.register_point_cloud("parabola_focus", [pv], radius=0.005, color=[1.0, 0.0, 0.0])

        # a = -0.2
        p = p_b1.copy()
        p[1:] = (1.0 - s)**2 * p_b1[1:] + 2 * (1 - s) * s * pv[1:] + s**2 * p_c1[1:]
        parabola_locus.append(p.copy())
        # ps.register_point_cloud("parabola_locus", parabola_locus, radius=0.002, color=[1.0, 1.0, 0.0])
        
        q = (1.0 - s) * q_b1 + s * q_c1
        return p, q, False
    
    return p_c1, q_c1, True

#region callback
def callback():
    global gripper_pos, rotvec, jaw_open, jaw_gap_open, jaw_gap_closed
    global tip_center_local, grasp_box, smooth, frame_id, frame_dt
    global fold_time, fold_error
    
    t = frame_id * frame_dt
    
    p, q, jaw_open = planned_pose_1(t)

    grip.set_pose(q, p - tip_center_local)

    grip.set_open(is_open=jaw_open, smooth=smooth, box=grasp_box, center_local=tip_center_local)

    psim.TextUnformatted(f"Grasped nodes = {grip.controlled}")

    # advance one physical step
    grip.step()
    
    if export:
        record_frame()
    
    # cam = ps.get_view_camera_parameters()
    # cam_pos = cam.get_position()
    # look_dir = cam.get_look_dir()
    
    # print(cam_pos, look_dir)
    
    #### error to final folded shape
    
    Xnow = cloth.positions.copy()

    E_fold = fold_error_to_final(
        Xnow,
        Xideal,
        side_a,
        side_b
    )

    fold_time.append(t)
    fold_error.append(E_fold)

    update_scene()
    
    if t >= T_end:
        cam_pos = [0.5803147554397583, 0.14937344193458557, 0.47480833530426025]
        target = [-0.7241836786270142, -0.4275602400302887, -0.5410640835762024]
        ps.look_at(cam_pos, target)
    
    # cam = ps.get_view_camera_parameters()
    # cam_pos = cam.get_position()
    # look_dir = cam.get_look_dir()
    # print(f'[{cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}]')
    # print(f'[{look_dir[0]}, {look_dir[1]}, {look_dir[2]}]')

    if t >= 11.0:
        final_error = E_fold
        print(f'final_error: {final_error}')
        ps.unshow()
    
    frame_id += 1
    
ps.set_user_callback(callback)
ps.show()

## The following part stacks the errors for each simulation. 

# comment the following for registering column two and onwards
# all_errors = np.asarray(fold_error).reshape(-1, 1) # (n, 1)
# np.savez(os.path.join(EXPORT_DIR, "all_errors.npz"), 
#          errors=np.asarray(all_errors, dtype=np.float64))

sys.path.insert(0, str(CLOTHILDE_ROOT))
errors_all = np.load(os.path.join(EXPORT_DIR, "all_errors.npz"))
all_errors = errors_all["errors"]
# comment the following once to start registering all_errors
all_errors = np.column_stack((all_errors, np.asarray(fold_error).reshape(-1, 1))) # (n, 8)

np.savez(os.path.join(EXPORT_DIR, "all_errors.npz"), 
         errors=np.asarray(all_errors, dtype=np.float64))

print(all_errors.shape)

if export:
    save_dynamic_data()
    print("Exported replay data to:", EXPORT_DIR)

errors_all = np.load(os.path.join(EXPORT_DIR, "all_errors.npz"))
all_errors = errors_all["errors"]

print(f'final_errors: {all_errors[-1, :]}')

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 15,
    "font.size": 20,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

plt.figure()
plt.plot(fold_time, all_errors, linewidth=1.3, label="simulation error")
plt.axhline(0.0, linestyle="--", linewidth=1, label="ideal fold")
plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Normalized RMS error to ideal fold $E(t)$")
# plt.title("Convergence to ideal 180° fold about x-axis")
plt.grid(True)
plt.legend(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
plt.tight_layout()
plt.savefig("error_curves.pdf", bbox_inches="tight")
# plt.show()
