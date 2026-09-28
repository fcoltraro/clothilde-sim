# To export a video of grasping

import sys, os
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
sys.path.append(parent_dir + "/python_code")
CLOTHILDE_ROOT = parent_dir + "/python_code"
EXPORT_DIR = CLOTHILDE_ROOT + "/exported_test_gripper_temp" # By keeping everything float64
# os.makedirs(EXPORT_DIR, exist_ok=True)
export = False

from implementation.Cloth0 import Cloth
from implementation.utils import createRectangularMesh

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

import numpy as np
import trimesh
import math

#region cloth
# na = 24
# nb = 20
na = 23
nb = 23

side_a = 0.8
side_b = 0.8

X, T = createRectangularMesh(a=side_a, b=side_b, na=na, nb=nb, h=0.0)
# X[:, 2] += 0.7
# X += 0.0001 * np.random.randn(X.shape[0], 3)

cloth = Cloth(X, T)
dt = cloth.estimateTimeStep(L=0.8)
# cloth.setSimulatorParameters(dt=dt, shr=0.1 * 0.0001, str=0.001 * 0.0001)
cloth.setSimulatorParameters(dt=1/60, sub_steps=6, slf=0.01)
# cloth.setSimulatorParameters(dt=1/60,sub_steps=8,mu_s=0.4,thck=1.05,kappa=1e-4)

grip1 = SimulateGripper(cloth)
grip2 = SimulateGripper(cloth)
# g = [grip1, grip2]

import polyscope as ps
import polyscope.imgui as psim

#region gripper

gripper1_p_history = []
gripper1_q_history = []
jaw_open1_history = []
controlled1_history = []

gripper2_p_history = []
gripper2_q_history = []
jaw_open2_history = []
controlled2_history = []

smooth=2 # displayed cloth surface may not be the raw vertex positions
# Hence while grasping, the cloth might not sit between the jaws

# initial pose controls 
gripper_pos1 = cloth.positions.mean(axis=0).copy()
gripper_pos1[2] = 0.5

gripper_pos2 = cloth.positions.mean(axis=0).copy()
gripper_pos2[0] -= 0.15
gripper_pos2[2] = 0.5

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

grip2.set_pose(q_init, gripper_pos2)
q02 = grip2.q.copy()
p02 = grip2.p.copy()

### helper functions
follow_offset = None
follow_enabled = False

# Gripper parallelopiped

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

# gripper 0
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
#gripper 1
ps.register_surface_mesh(
    "gripper_base2",
    transform_mesh(V_base0, q02, p02),
    F_base, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_left2",
    transform_mesh(V_left0, q02, p02, local_offset=np.array([-jaw_gap_open/2, 0, 0])),
    F_left, color=YELLOW
)
ps.register_surface_mesh(
    "gripper_right2",
    transform_mesh(V_right0, q02, p02, local_offset=np.array([ jaw_gap_open/2, 0, 0])),
    F_right, color=YELLOW
)

# grasp box
grasp_box = 0.001 * np.array([30, 30, 30], dtype=float)
tip_center_local = 0.001 * np.array([0.0, 0.0, - (52 - grasp_box[2] * 1000 / 2)], dtype=float)

# grasp_box = 0.001 * np.array([6, 30, 6], dtype=float)
# tip_center_local = 0.001 * np.array([0.0, 0.0, -49], dtype=float)

# 37 - 3 + 15 mm (check CAD file)  is the distance of the jaw extensions 
# from the origin of the gripper frame

# V_dbg = get_box_vertices_world_offset(p=g[0].p, q=g[0].q, box_size=grasp_box, center_local=tip_center_local)

# ps.register_surface_mesh(
#     "grasp_box",
#     V_dbg,
#     box_faces,
#     color=[1.0, 1.0, 0.0],
#     transparency=0.55,
#     material="wax"
# )

# initialize grasped nodes
# ps.register_point_cloud("grasped_nodes", np.zeros((0,3)), radius=0.002, color=[1.0, 0.0, 0.0])

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

    q2 = gripper2_q_history[k]
    p2 = gripper2_p_history[k]
    jaw_open2 = jaw_open2_history[k]
    
    current_gap1 = jaw_gap_open if jaw_open1 else jaw_gap_closed
    current_gap2 = jaw_gap_open if jaw_open2 else jaw_gap_closed

    ps.get_surface_mesh("gripper_base1").update_vertex_positions(
        transform_mesh(V_base0, q1, p1)
    )
    ps.get_surface_mesh("gripper_left1").update_vertex_positions(
        transform_mesh(V_left0, q1, p1, local_offset=np.array([-current_gap1/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right1").update_vertex_positions(
        transform_mesh(V_right0, q1, p1, local_offset=np.array([ current_gap1/2, 0, 0]))
    )
    
    ps.get_surface_mesh("gripper_base2").update_vertex_positions(
        transform_mesh(V_base0, q2, p2)
    )
    ps.get_surface_mesh("gripper_left2").update_vertex_positions(
        transform_mesh(V_left0, q2, p2, local_offset=np.array([-current_gap2/2, 0, 0]))
    )
    ps.get_surface_mesh("gripper_right2").update_vertex_positions(
        transform_mesh(V_right0, q2, p2, local_offset=np.array([ current_gap2/2, 0, 0]))
    )

    # update grasp box
    # V_dbg = get_box_vertices_world_offset(g[0].p, g[0].q, grasp_box, tip_center_local)
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
    psim.TextUnformatted(f"Gripper 1 nodes = {ctrl1}")
    if ctrl1.size > 0:
        ps.register_point_cloud("grasped_nodes1", phi_all[ctrl1], radius=0.002, color=[1.0, 0.0, 0.0])
        # ps.get_point_cloud("grasped_nodes").update_point_positions(phi_all[ctrl])
    else:
        ps.register_point_cloud("grasped_nodes1", np.zeros((0,3)), radius=0.002, color=[1.0, 0.0, 0.0])

    ctrl2 = np.asarray(controlled2_history[k], dtype=int)
    psim.TextUnformatted(f"Gripper 2 nodes = {ctrl2}")
    if ctrl2.size > 0:
        ps.register_point_cloud("grasped_nodes2", phi_all[ctrl2], radius=0.002, color=[1.0, 0.0, 0.0])
        # ps.get_point_cloud("grasped_nodes").update_point_positions(phi_all[ctrl])
    else:
        ps.register_point_cloud("grasped_nodes2", np.zeros((0,3)), radius=0.002, color=[1.0, 0.0, 0.0])

    

#region folding
# called every frame: interactive frame loop
# step(): one physics step per frame
frame_dt = cloth.frame_rate

p_a1 = p01.copy()
q_a1 = q01.copy()

p_a2 = p02.copy()
q_a2 = q02.copy()

# box; no rotation; 2 nodes: 527, 528
p_b1 = cloth.positions[len(cloth.positions) - 1] + [- 1/8 * (side_a / na), 3/4 * (side_b / nb), 0] 
q_b1 = quat_mul(quat_from_axis_angle([0, 0, 1], -np.pi/2), 
                        quat_from_axis_angle([0, 1, 0], -np.pi/4))
# q_b1 = quat_from_axis_angle([0, 0, 1], -np.pi/2) # no rotation

p_b2 = p_a2.copy()
q_b2 = q_a2.copy()
    
p_c1 = np.array([side_a/2, -side_b/2, 0.2])
q_c1 = quat_mul(q_b1.copy(), quat_from_axis_angle([0, 1, 0], np.pi/4))

p_c2 = p_b2.copy()
q_c2 = q_b2.copy()

p_d1 = None
q_d1 = q01.copy()

p_d2 = None
q_d2 = q_c2.copy()

p_e1 = np.array([-side_a/2 * 0.8, -side_b/2, 0.2])
q_e1 = q01.copy()

p_e2 = np.array([-side_a/2 * 0.8, 0, 0.2])
q_e2 = q_d2.copy()

p_f1 = p_e1.copy()
q_f1 = q_e1.copy()

p_f2 = p_e2.copy()
q_f2 = q_e2.copy()

p_g1 = None
q_g1 = q_f1.copy()

p_g2 = None
q_g2 = q_f1.copy()

p_h1 = np.array([0, -side_b/2, 0.8])
q_h1 = q_g1.copy()

p_h2 = np.array([0, side_b/2, 0.8])
q_h2 = q_g2.copy()

p_i1 = None
q_i1 = q_h1.copy()

p_i2 = None
q_i2 = q_h2.copy()



T1 = 2.0   # seconds: stay at initial pose, gripper open
T2 = 1.0   # move from p_start, q_start to one corner of cloth
T3 = 1.0   # close gripper
T4 = 3.0   # parabolic motion (follows a quadratic Bezier curve)
T_second_start = T1 + T2 + T3 + T4
T5 = 2.0
T6 = 1.0
T7 = 3.0
T8 = 1.0
T_third_start = T_second_start + T5 + T6 + T7 + T8
T_cam = 3.0
T9 = 2.0
T10 = 1.0
T11 = 5.0
T_fourth_start = T_third_start + T_cam + T9 + T10 + T11
T_cam2 = 2.0
T12 = 5.0
T13 = 1.0
T_end = T_fourth_start + T_cam2 + T12 + T13 + 0.5  # small hold after motion

def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)
# standard smoothing function: that gives f'(0) = f'(1) = 0

parabola_locus = []

#region planned pose
def planned_pose_1(t):
    # First fold
    global p_d1, p_e1, p_g1, p_i1
    if t < T1:
        return p_a1, q_a1, True

    if t < T1 + T2:
        tau = (t - T1) / T2 # varies from 0 to 1
        s = smoothstep(tau)

        p = (1.0 - s) * p_a1 + s * p_b1
        q = (1.0 - s) * q_a1 + s * q_b1

        return p, q, True
    
    if t < T1 + T2 + T3:
        return p_b1, q_b1, False
    
    if t < T1 + T2 + T3 + T4:
        tau = (t - T1 - T2 - T3) / T4
        s = smoothstep(tau)
        
        pv = np.array([side_a/2, side_b/2 - 0.05, 0.15]) # control handle to pull the parabola
        # ps.register_point_cloud("parabola_focus", [pv], radius=0.005, color=[1.0, 0.0, 0.0])

        # a = -0.2
        p = p_b1.copy()
        p[1:] = (1.0 - s)**2 * p_b1[1:] + 2 * (1 - s) * s * pv[1:] + s**2 * p_c1[1:]
        parabola_locus.append(p.copy())
        # ps.register_point_cloud("parabola_locus", parabola_locus, radius=0.002, color=[1.0, 1.0, 0.0])
        
        q = (1.0 - s) * q_b1 + s * q_c1
        return p, q, False
    
    # second fold
    if t < T_second_start + T5:
        tau = (t - T_second_start) / T5 # varies from 0 to 1
        s = smoothstep(tau) 

        p_d1 = (cloth.positions[(na * nb) - 1].copy() + cloth.positions[na - 1].copy()) / 2
        # p_e1 = p_d1 + np.array([0, 0, 0.2])
        
        p = (1.0 - s) * p_c1 + s * p_d1
        q = (1.0 - s) * q_c1 + s * q_d1

        return p, q, True
    
    if t < T_second_start + T5 + T6:
        return p_d1, q_d1, False
    
    if t < T_second_start + T5 + T6 + T7:
        tau = (t - T_second_start - T5 - T6) / T7
        s = smoothstep(tau)
        
        # p = (1.0 - s) * p_d1 + s * p_e1
        q = (1.0 - s) * q_d1 + s * q_e1
        
        pde1 = np.array([side_a/2 - 0.05, side_b/2, 0.2]) # control handle to pull the parabola

        p = p_d1.copy()
        p[0] = (1.0 - s)**2 * p_d1[0] + 2 * (1 - s) * s * pde1[0] + s**2 * p_e1[0]
        p[2] = (1.0 - s)**2 * p_d1[2] + 2 * (1 - s) * s * pde1[2] + s**2 * p_e1[2]
        
        q = (1.0 - s) * q_d1 + s * q_e1

        return p, q, False
    
    if t < T_second_start + T5 + T6 + T7 + T8:
        return p_f1, q_f1, True
    
    # Lift
    if t < T_third_start + T_cam + T9 :
        tau = (t - T_third_start - T_cam) / T9 # varies from 0 to 1
        s = smoothstep(tau) 

        p_g1 = cloth.positions[na - 1].copy()
        # p_e1 = p_d1 + np.array([0, 0, 0.2])
        
        p = (1.0 - s) * p_f1 + s * p_g1
        q = (1.0 - s) * q_f1 + s * q_g1

        return p, q, True
    
    if t < T_third_start + T_cam + T9 + T10:
        return p_g1, q_g1, False
    
    if t < T_third_start + T_cam + T9 + T10 + T11:
        tau = (t - T_third_start - T_cam - T9 - T10) / T11 # varies from 0 to 1
        s = smoothstep(tau) 
        
        p = (1.0 - s) * p_g1 + s * p_h1
        q = (1.0 - s) * q_g1 + s * q_h1

        return p, q, False
    
    # spiral motion
    if t < T_fourth_start + T_cam2 + T12:
        tau = (t - T_fourth_start - T_cam2) / T12 # varies from 0 to 1
        s = smoothstep(tau) 
        
        r0 = side_b / 2
        r1 = side_b / 4
        r = (1.0 - s) * r0 + s * r1 # conin spiral
        
        theta0 = -np.pi / 2     # because start point is [0, -r, z]
        n_turns = 2.0           # number of revolutions
        spiral_height = 0.3
        
        theta = theta0 + 2.0 * np.pi * n_turns * s
        
        p_i1 = np.array([r * np.cos(theta), r * np.sin(theta), p_h1[2] + spiral_height * s])

        return p_i1, q_i1, False
    
    if t < T_fourth_start + T_cam2 + T12 + T13:
        return p_i1, q_i1, True
        
    return p_i1, q_i1, True

#region plan-pose 2
def planned_pose_2(t):
    # Second fold
    global p_d2, p_e2, p_g2, p_i2
    if t < T_second_start:
        return p_a2, q_a2, True
    
    if t < T_second_start + T5:
        tau = (t - T_second_start) / T5 # varies from 0 to 1
        s = smoothstep(tau) 

        p_d2 = cloth.positions[(na * math.ceil(nb / 2)) - 1].copy()
        # p_e2 = p_d2 + np.array([0, 0, 0.2])
        
        p = (1.0 - s) * p_c2 + s * p_d2
        q = (1.0 - s) * q_c2 + s * q_d2

        return p, q, True
    
    if t < T_second_start + T5 + T6:
        return p_d2, q_d2, False

    # Lift
    if t < T_second_start + T5 + T6 + T7:
        tau = (t - T_second_start - T5 - T6) / T7
        s = smoothstep(tau)

        pde2 = np.array([side_a/2 - 0.05, - side_b/2, 0.2]) # control handle to pull the parabola

        p = p_d2.copy()
        p[0] = (1.0 - s)**2 * p_d2[0] + 2 * (1 - s) * s * pde2[0] + s**2 * p_e2[0]
        p[2] = (1.0 - s)**2 * p_d2[2] + 2 * (1 - s) * s * pde2[2] + s**2 * p_e2[2]
        
        q = (1.0 - s) * q_d2 + s * q_e2

        return p, q, False
    
    if t < T_second_start + T5 + T6 + T7 + T8:
        return p_f2, q_f2, True
    
    if t < T_third_start + T_cam + T9 :
        tau = (t - T_third_start - T_cam) / T9 # varies from 0 to 1
        s = smoothstep(tau) 

        p_g2 = cloth.positions[(na * nb) - 1].copy()
        # p_e1 = p_d1 + np.array([0, 0, 0.2])
        
        p = (1.0 - s) * p_f2 + s * p_g2
        q = (1.0 - s) * q_f2 + s * q_g2

        return p, q, True
    
    if t < T_third_start + T_cam + T9 + T10:
        return p_g2, q_g2, False

    if t < T_third_start + T_cam + T9 + T10 + T11:
        tau = (t - T_third_start - T_cam - T9 - T10) / T11 # varies from 0 to 1
        s = smoothstep(tau) 
        
        p = (1.0 - s) * p_g2 + s * p_h2
        q = (1.0 - s) * q_g2 + s * q_h2

        return p, q, False
    
    # spiral
    if t < T_fourth_start + T_cam2 + T12:
        tau = (t - T_fourth_start - T_cam2) / T12 # varies from 0 to 1
        s = smoothstep(tau) 
        
        r0 = side_b / 2
        r1 = side_b / 4
        r = (1.0 - s) * r0 + s * r1 # conin spiral
        
        theta0 = np.pi / 2     # because start point is [0, -r, z]
        n_turns = 2.0           # number of revolutions
        spiral_height = 0.3
        
        theta = theta0 + 2.0 * np.pi * n_turns * s
        
        p_i2 = np.array([r * np.cos(theta), r * np.sin(theta), p_h2[2] + spiral_height * s])

        return p_i2, q_i2, False
    
    if t < T_fourth_start + T_cam2 + T12 + T13:
        return p_i2, q_i2, True
        
    return p_i2, q_i2, True

#region one frame
def simulate_one_frame(t):
    # global gripper_pos, rotvec, jaw_open, jaw_gap_open, jaw_gap_closed
    # global tip_center_local, grasp_box, smooth, frame_id, frame_dt
        
    p1, q1, jaw_open1 = planned_pose_1(t)
    p2, q2, jaw_open2 = planned_pose_2(t)

    grip1.set_pose(q1, p1 - tip_center_local)
    grip2.set_pose(q2, p2 - tip_center_local)

    grip1.set_open(is_open=jaw_open1, smooth=smooth, box=grasp_box, center_local=tip_center_local)
    grip2.set_open(is_open=jaw_open2, smooth=smooth, box=grasp_box, center_local=tip_center_local)


    # advance one physical step
    grip1.step(grippers=[grip1, grip2])
    
    gripper1_p_history.append(grip1.p.copy())
    gripper1_q_history.append(grip1.q.copy())
    jaw_open1_history.append(bool(jaw_open1))
    controlled1_history.append(list(grip1.controlled))
    
    gripper2_p_history.append(grip2.p.copy())
    gripper2_q_history.append(grip2.q.copy())
    jaw_open2_history.append(bool(jaw_open2))
    controlled2_history.append(list(grip2.controlled))
    
frame_id = 0
n_frames = int(T_end / frame_dt) + 1

while frame_id * frame_dt <= T_end:
    t = frame_id * frame_dt
    simulate_one_frame(t)
    
    if frame_id % 30 == 0:
        print(f"simulated frame {frame_id}/{n_frames}")
    frame_id += 1
    

#region callback

movie_frame = 0
movie_speed = 2
repeat_movie = True

cam_pos0 = [1.5683601, 0.6266863, 0.4967863]
target =   [-0.8907767, -0.43488115, -0.13189112]
up_dir = [-0.11852091, -0.05786243, 0.9912642 ]

camera_t1 = T_second_start
camera_t2 = T_third_start
camera_t3 = T_fourth_start

def movie_callback():
    global movie_frame
    update_scene(movie_frame)
    
    t = movie_frame * frame_dt
    if t <= camera_t1:
        ps.look_at_dir(cam_pos0, target, up_dir)
    elif t <= camera_t2:
        cam_pos1 = [-1.2709341, 1.019664, 0.6839701] 
        target1 = [0.71897674, -0.6090666, -0.33482885] 
        up_dir1 = [0.25548074, -0.21642536, 0.942279]
        ps.look_at_dir(cam_pos1, target1, up_dir1, fly_to=True)
    elif t <= camera_t2 + T_cam:
        cam_pos2 = [-0.88652235, -1.4744945, 0.64292073] 
        target2 = [ 0.5006195, 0.80767655, -0.31151044] 
        up_dir2 = [0.16411406, 0.2647741, 0.95024276]
        ps.look_at_dir(cam_pos2, target2, up_dir2, fly_to=True)
    elif t <= camera_t3 + T_cam2:
        cam_pos3 = [-1.5557017, -1.0377461, 0.96404564] 
        target3 =[ 0.8185977, 0.50575316, -0.2722345 ] 
        up_dir3 =[0.23159777, 0.14308776, 0.962231  ]
        ps.look_at_dir(cam_pos3, target3, up_dir3, fly_to=True)
    elif t <= T_fourth_start + T_cam2 + T12:
        cam_pos4 = [-2.1147923, -1.549536, 1.477028]
        target4 = [0.7192589, 0.51037717, -0.47136176]
        up_dir4 = [0.38441518, 0.27277625, 0.88194]
        ps.look_at_dir(cam_pos4, target4, up_dir4, fly_to=True)
          
    movie_frame += movie_speed

    if movie_frame >= min(len(cloth.history_pos), len(gripper1_p_history)):
        if repeat_movie:
            movie_frame = 0
        else:
            movie_frame = min(len(cloth.history_pos), len(gripper1_p_history)) - 1
            update_scene(movie_frame)
            ps.clear_user_callback()
    
ps.set_user_callback(movie_callback)
ps.show()
cam = ps.get_view_camera_parameters()
cam_pos = cam.get_position()
look_dir = cam.get_look_dir()
up_dir = cam.get_up_dir()

print(cam_pos, look_dir, up_dir)
ps.clear_user_callback()