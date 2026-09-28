# To export a video of grasping

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

is_gripper_box = False # True: box, False: square pyramid

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
import math
import time

grasp_time = 0.0
total_time = 0.0

#region cloth
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
# cloth.setSimulatorParameters(dt=dt, shr=0.1 * 0.0001, str=0.001 * 0.0001)
# cloth.setSimulatorParameters(dt=1/60, sub_steps=6, slf=0.01)

# ## For simulation
# cloth.setSimulatorParameters(dt = 1/60, tol = 0.0075, sub_steps = 6,
#                                rho = 0.3046, delta = 0.139, alpha = 0.416,
#                                kappa = 0.25*1e-4, kappa_bnd = 0, 
#                                str = 0.01*1e-4, shr = 5*1e-4, slf = 1*1e-4,
#                                mu_f = 0.2, mu_s = 0.6, thck = 0.95, max_mov= 0.1)

## For Franka experiment
cloth.setSimulatorParameters(dt = 1/60, tol = 0.0075, sub_steps = 6,
                               rho = 0.3046, delta = 0.139, alpha = 0.416,
                               kappa = 0.40*1e-4, kappa_bnd = 0, 
                               str = 0.01*1e-4, shr = 5*1e-4, slf = 1*1e-4,
                               mu_f = 0.40, mu_s = 1.3, thck = 0.95, max_mov= 0.1)


## To verify the real cloth behavior
# cloth.setSimulatorParameters(dt = 1/60, tol = 0.0075, sub_steps = 6,
#                                rho = 0.3046, delta = 0.139, alpha = 0.416,
#                                kappa = 1*1e-4, kappa_bnd = 0, 
#                                str = 0.01*1e-4, shr = 5*1e-4, slf = 1*1e-4,
#                                mu_f = 0.2, mu_s = 1.5, thck = 0.95, max_mov= 0.1)

grip1 = SimulateGripper(cloth)

import polyscope as ps
import polyscope.imgui as psim

#region gripper
gripper1_p_history = []
gripper1_q_history = []
jaw_open1_history = []
controlled1_history = []

smooth=2 # displayed cloth surface may not be the raw vertex positions
# Hence while grasping, the cloth might not sit between the jaws

# initial pose controls 
gripper_pos1 = cloth.positions.mean(axis=0).copy()
gripper_pos1[2] = 0.5

jaw_open = True
jaw_gap_open = 0.0
jaw_gap_closed = -0.001 * (30 - 6)
# 30 mm - 6 mm (extension length) is the gap between the grippers in my CAD model
YELLOW = [186/256, 142/256, 35/256]

rotvec = np.array([0.0, 0.0, 0.0], dtype=float)

q_init = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0)
grip1.set_pose(q_init, gripper_pos1)
q01 = grip1.q.copy()
p01 = grip1.p.copy()

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

## gripper 0
ps.register_surface_mesh(
    "gripper_base1",
    transform_mesh(V_base0, q01, p01),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left1",
    transform_mesh(V_left0, q01, p01, local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right1",
    transform_mesh(V_right0, q01, p01, local_offset=np.array([ jaw_gap_open/2, 0, 0])),
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

# V_dbg = get_box_vertices_world_offset(p=grip1.p, q=grip1.q, box_size=grasp_box, center_local=tip_center_local)

# ps.register_surface_mesh(
#     "grasp_box",
#     V_dbg,
#     box_faces,
#     color=[1.0, 1.0, 0.0],
#     transparency=0.3,
#     material="wax"
# )

# initialize grasped nodes
# ps.register_point_cloud("grasped_nodes", np.zeros((0,3)), radius=0.004, color=[1.0, 0.0, 0.0])

#region update scene
# only for visualization (redraw/update): also called every frame through callback
  
def update_scene(k):
    # copied from Cloth.py: to update meshes
    phi_mat = cloth.history_pos[k]
    phi_all = cloth.Am @ phi_mat
    for _ in range(smooth):
        phi_all = cloth.S @ phi_all
        

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)

    q1 = gripper1_q_history[k]
    p1 = gripper1_p_history[k]
    jaw_open1 = jaw_open1_history[k]
    
    current_gap1 = jaw_gap_open if jaw_open1 else jaw_gap_closed

    ps.get_surface_mesh("gripper_base1").update_vertex_positions(
        transform_mesh(V_base0, q1, p1)
    )
    ps.get_surface_mesh("gripper_left1").update_vertex_positions(
        transform_mesh(V_left0, q1, p1, local_offset=np.array([-current_gap1/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right1").update_vertex_positions(
        transform_mesh(V_right0, q1, p1, local_offset=np.array([ current_gap1/2, 0, 0]))
    )

    # update grasp box
    # V_dbg = get_box_vertices_world_offset(p1, q1, grasp_box, tip_center_local)
    # ps.get_surface_mesh("grasp_box").update_vertex_positions(V_dbg)

    # if jaw_open:
    #     ps.get_surface_mesh("grasp_box").set_color([0.0, 1.0, 0.0])
    # else:
    #     ps.get_surface_mesh("grasp_box").set_color([1.0, 0.0, 0.0])

    # update gripper frame from current pose
    R = quat_to_rotmat(grip1.q)
    try:
        ps.remove_structure("gripper_frame_live")
    except:
        pass
    
    ctrl1 = np.asarray(controlled1_history[k], dtype=int)
    
    # psim.TextUnformatted(f"Grasped nodes = {ctrl1}")
    # if ctrl1.size > 0:
    #     ps.register_point_cloud("grasped_nodes1", phi_all[ctrl1], radius=0.004, color=[1.0, 0.0, 0.0])
    #     # ps.get_point_cloud("grasped_nodes").update_point_positions(phi_all[ctrl])
    # else:
    #     ps.register_point_cloud("grasped_nodes1", np.zeros((0,3)), radius=0.004, color=[1.0, 0.0, 0.0])


#region folding
# called every frame: interactive frame loop
# step(): one physics step per frame
frame_dt = cloth.frame_rate

p_a1 = p01.copy()
q_a1 = q01.copy()

# # box; no rotation; 2 nodes: 527, 528
# p_b1 = cloth.positions[len(cloth.positions) - 1] + [- 1/8 * (side_a / na), 3/4 * (side_b / nb), 0] 
# q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
#                         quat_from_axis_angle([0, 1, 0], -np.pi/4))
# # q_b1 = quat_from_axis_angle([0, 0, 1], -np.pi/2) # no rotation

#region cases
#### Cases
case = 0
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

# if is_rotate_gripper:
#     q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
#                         quat_from_axis_angle([0, 1, 0], -np.pi/4))
# else:
#     q_b1 = quat_from_axis_angle([0, 0, 1], -np.pi/2) # no rotation
#     # q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
#     #                     quat_from_axis_angle([0, 1, 0], -np.pi/4))

## This rotation to match that of the Franka simulation. NOTE: Without squeeze as the 
## corner is placed between the gripper in real robot.
if is_rotate_gripper:
    q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], 0), 
                        quat_from_axis_angle([1, 0, 0], -np.pi/4))
else:
    q_b1 = quat_from_axis_angle([0, 0, 1], 0) # no rotation
    # q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
    #                     quat_from_axis_angle([0, 1, 0], -np.pi/4))

# p_c1 = np.array([side_a/2, -side_b/2, 0.04]) # simulation
p_c1 = np.array([side_a/2, -side_b/2, 0.07]) # Franka

if is_rotate_gripper:
    # q_c1 = quat_mul(q_b1.copy(), quat_from_axis_angle([0, 1, 0], np.pi/4))
    q_c1 = quat_mul(q_b1.copy(), quat_from_axis_angle([1, 0, 0], np.pi/4)) # To match Franka
else:
    q_c1 = q_b1.copy()

# T1 = 2.0   # seconds: stay at initial pose, gripper open
# T2 = 1.0   # move from p_start, q_start to one corner of cloth
# T3 = 1.0   # close gripper
# T4 = 3.0   # parabolic motion (follows a quadratic Bezier curve)
# T_cam1 = 3.0
# # T_cam2 = 2.0
# # T_cam3 = 2.0

## Start directly from a corner
T1 = 1.0   # seconds: stay at initial pose, gripper open
T2 = 1.0   # close gripper
T3 = 3.0   # parabolic motion (follows a quadratic Bezier curve)
# T_cam1 = 3.0

# T_end = T1 + T2 + T3 + T4 + T_cam1 + 0.5  # small hold after motion

T_end = T1 + T2 + T3 + 3.0  # small hold after motion

def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s) # Used for Franka
    # return 10 * s**3 - 15 * s**4 + 6 * s**5 # Simulation
    # return s
# standard smoothing function: that gives f'(0) = f'(1) = 0

parabola_locus = []

#region planned pose
# def planned_pose_1(t):
#     # First fold
#     global p_d1, p_e1, p_g1, p_i1
#     if t < T1:
#         return p_a1, q_a1, True

#     if t < T1 + T2:
#         tau = (t - T1) / T2 # varies from 0 to 1
#         s = smoothstep(tau)

#         p = (1.0 - s) * p_a1 + s * p_b1
#         q = (1.0 - s) * q_a1 + s * q_b1

#         return p, q, True
    
#     if t < T1 + T2 + T3:
#         return p_b1, q_b1, False
    
#     if t < T1 + T2 + T3 + T4:
#         tau = (t - T1 - T2 - T3) / T4
#         s = smoothstep(tau)
        
#         pv = np.array([side_a/2, side_b/2 - 0.05, 0.15]) # control handle to pull the parabola
#         # ps.register_point_cloud("parabola_focus", [pv], radius=0.005, color=[1.0, 0.0, 0.0])

#         # a = -0.2
#         p = p_b1.copy()
#         p[1:] = (1.0 - s)**2 * p_b1[1:] + 2 * (1 - s) * s * pv[1:] + s**2 * p_c1[1:]
#         parabola_locus.append(p.copy())
#         # ps.register_point_cloud("parabola_locus", parabola_locus, radius=0.002, color=[1.0, 1.0, 0.0])
        
#         q = (1.0 - s) * q_b1 + s * q_c1
#         return p, q, False
    
#     return p_c1, q_c1, True

def planned_pose_1(t):
    # First fold
    if t < T1:
        return p_b1, q_b1, True
    
    if t < T1 + T2:
        return p_b1, q_b1, False
    
    if t < T1 + T2 + T3:
        tau = (t - T1 - T2) / T3
        s = smoothstep(tau)
        
        # control point to pull the parabola
        # pv = np.array([side_a/2, side_b/2, 0.18]) # Simulation

        pv = np.array([side_a/2, side_b/2 - 2*side_b/nb, 0.13]) # Franka

        # ps.register_point_cloud("parabola_focus", [pv], radius=0.005, color=[1.0, 0.0, 0.0])

        # a = -0.2
        p = p_b1.copy()
        p[1:] = (1.0 - s)**2 * p_b1[1:] + 2 * (1 - s) * s * pv[1:] + s**2 * p_c1[1:]
        parabola_locus.append(p.copy())
        # ps.register_point_cloud("parabola_locus", parabola_locus, radius=0.002, color=[1.0, 1.0, 0.0])
        
        q = (1.0 - s) * q_b1 + s * q_c1
        return p, q, False
    
    return p_c1, q_c1, True

#region one frame
def simulate_one_frame(t):
    # global gripper_pos, rotvec, jaw_open, jaw_gap_open, jaw_gap_closed
    # global tip_center_local, grasp_box, smooth, frame_id, frame_dt
    global grasp_time, total_time
       
    t_frame = time.perf_counter()
    
    p1, q1, jaw_open1 = planned_pose_1(t)
    
    t_grasp = time.perf_counter()

    # p1 - tip_center_local because set_pose() needs position wrt to the gripper CAD frame
    grip1.set_pose(q1, p1 - tip_center_local)

    grip1.set_open(is_open=jaw_open1, smooth=smooth, box=grasp_box, center_local=tip_center_local, squeeze_enabled=True)

    grasp_time += time.perf_counter() - t_grasp
    
    # advance one physical step: cloth physics + grasp constraints
    grip1.step(grippers=[grip1])
    
    total_time += time.perf_counter() - t_frame
    
    gripper1_p_history.append(grip1.p.copy())
    gripper1_q_history.append(grip1.q.copy())
    jaw_open1_history.append(bool(jaw_open1))
    controlled1_history.append(list(grip1.controlled))
    
frame_id = 0
n_frames = int(T_end / frame_dt) + 1

while frame_id * frame_dt <= T_end:
    t = frame_id * frame_dt
    simulate_one_frame(t)
    
    if frame_id % 30 == 0:
        print(f"simulated frame {frame_id}/{n_frames}")
    frame_id += 1

# print time (for ICINCO paper after the review)

print(f"Total simulation time: {total_time:.4f} s")
print(f"Gripper-interface time: {grasp_time:.4f} s")
print(f"Gripper-interface percentage: {100*grasp_time/total_time:.3f} %")
print(f"Average interface time/frame: {1e3*grasp_time/frame_id:.4f} ms")
    
#region callback
movie_frame = 0
movie_speed = 1
repeat_movie = True
# cam_pos0 = [1.5926634, 0.6104875, 0.5650971]
# target =   [-0.793475,   -0.5274152,  -0.30369517]
# up_dir = [-0.2529201,  -0.16811357,  0.9527693 ]

## For the ICINCO paper
cam_pos0 = [0.7670964002609253, 0.23198041319847107, 0.6202637553215027]
target = [-0.7449334263801575, -0.3294577896595001, -0.5801136493682861]
up_dir = [-0.5305428504943848, -0.23464040458202362, 0.8145356178283691]

camera_t1 = T1 + T2 + T3
# camera_t1 = T1 + T2 + T3 + T4

def movie_callback():
    global movie_frame
    update_scene(movie_frame)
    
    t = movie_frame * frame_dt
    # if t <= camera_t1:
    #     ps.look_at_dir(cam_pos0, target, up_dir)
    #     # For ICINCO paper
    #     ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    #     ps.set_shadow_darkness(0.7)
    #     ps.set_shadow_blur_iters(6)
    #     ps.set_ground_plane_height_mode('manual')
    #     ps.set_ground_plane_height(-0.005)

    # elif t <= camera_t1 + T_cam1:
    #     cam_pos1 = [0.9204815626144409, -1.1966215372085571, 1.1952921152114868]
    #     target1 = [-0.5405255556106567, 0.5399496555328369, -0.64520263671875]
    #     up_dir1 = [-0.4564702808856964, 0.455983966588974, 0.7640115022659302]
    #     ps.look_at_dir(cam_pos1, target1, up_dir1, fly_to=True)

    movie_frame += movie_speed

    if movie_frame >= min(len(cloth.history_pos), len(gripper1_p_history)):
        if repeat_movie:
            movie_frame = 0
        else:
            movie_frame = min(len(cloth.history_pos), len(gripper1_p_history)) - 1
            update_scene(movie_frame)
            ps.clear_user_callback()
   
#region export
if export:       
    n = len(gripper1_p_history)
    export_times = np.arange(n, dtype=np.float64) * frame_dt
    
    ## To change the position to the grasp_box center
    tip_position = np.array([
    gripper1_p_history[i] + quat_rotate_vector(gripper1_q_history[i], tip_center_local)
    for i in range(len(gripper1_p_history))
    ], dtype=np.float64)
    
    np.savez(os.path.join(EXPORT_DIR, "gripper_poses.npz"),
                t=np.asarray(export_times, dtype=np.float64),
                position=np.asarray(tip_position, dtype=np.float64),
                quaternion=np.asarray(gripper1_q_history, dtype=np.float64),
                jaw_status=np.array(jaw_open1_history)
                )
    print(f'gripper poses exported to gripper_poses.npz')
    
    ## everything expressed in Franka frame 
    ## T_F_C: Cloth frame expressed in Franka frame (transform F to match C)
    q_CinF = quat_from_axis_angle([0, 0, 1], np.pi)
    t_CinF = [0.45 + side_a/2, 0, 0]
    
    positions_F = []
    quats_F = []
    
    for p_GinC, q_GinC in zip(tip_position, gripper1_q_history):
        p_GinF = quat_rotate_vector(q_CinF, p_GinC) + t_CinF
        q_GinF = quat_normalize(quat_mul(q_GinC, q_CinF))
        
        q_GinF = quat_normalize(quat_mul(quat_from_axis_angle([0, 1, 0], np.pi), q_GinF))
        # The above is to correct for the orientation mismatch between the Franka base frame orientation
        # and the gripper frame orientation
         
        positions_F.append(p_GinF)
        quats_F.append(q_GinF)

    positions_F = np.asarray(positions_F, dtype=np.float64)
    quats_F = np.asarray(quats_F, dtype=np.float64) 
    
    np.savez(os.path.join(EXPORT_DIR, "gripper_poses_Franka.npz"),
            t=np.asarray(export_times, dtype=np.float64),
            position=np.asarray(positions_F, dtype=np.float64),
            quaternion=np.asarray(quats_F, dtype=np.float64),
            jaw_status=np.array(jaw_open1_history)
            )
    print(f'gripper poses in Franka frame exported to gripper_poses_Franka.npz') 
        
ps.set_user_callback(movie_callback)
ps.show()
cam = ps.get_view_camera_parameters()
cam_pos = cam.get_position()
look_dir = cam.get_look_dir()
up_dir = cam.get_up_dir()

print(f'[{cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]}]')
print(f'[{look_dir[0]}, {look_dir[1]}, {look_dir[2]}]')
print(f'[{up_dir[0]}, {up_dir[1]}, {up_dir[2]}]')
ps.clear_user_callback()

# if not export:
#     def read_gripper_poses(export_dir):
#         gripper_poses = np.load(os.path.join(export_dir, "gripper_poses_Franka.npz"))
#         times = gripper_poses["t"]
#         p_gripper = gripper_poses["position"]
#         q_gripper = gripper_poses["quaternion"]
#         jaw_status = gripper_poses["jaw_status"].astype(int)
#         return p_gripper, q_gripper, jaw_status, times

#     p_gripper, q_gripper, jaw_status, times = read_gripper_poses(EXPORT_DIR)
#     print(f'gripper quaternion: {q_gripper[0]}')
