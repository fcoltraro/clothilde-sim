# Cloth.py handles all the physics
# This file provides kinematic boundary condition on some grasped nodes

import numpy as np
from implementation.Cloth import Cloth

bbox_edges = np.array([
    [0,1], [1,2], [2,3], [3,0],   # bottom
    [4,5], [5,6], [6,7], [7,4],   # top
    [0,4], [1,5], [2,6], [3,7],   # verticals
], dtype=int)

def make_aabb_vertices_local(face_min, face_max):
    xmin, ymin, zmin = face_min
    xmax, ymax, zmax = face_max

    return np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=float)

# =========================
# Quaternion utilities
# =========================
def quat_normalize(q):
    q = np.array(q, dtype=float).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_conjugate(q):
    q = np.array(q, dtype=float).reshape(4,)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(q1, q2):
    q1 = np.array(q1, dtype=float).reshape(4,)
    q2 = np.array(q2, dtype=float).reshape(4,)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=float).reshape(3,)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    s = np.sin(half)
    return quat_normalize(
        np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)
    )

# rotation: q * vec in quat form * q^{-1}
def quat_rotate_vector(q, v):
    q = quat_normalize(q)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, vq), quat_conjugate(q))[1:]

def quat_rotate_points(q, X):
    X = np.array(X, dtype=float)
    return np.vstack([quat_rotate_vector(q, x) for x in X])

def quat_transform_points(p, q, X):
    X = np.array(X, dtype=float)
    return np.vstack([
        quat_rotate_vector(q, x) + p
        for x in X
    ])

def quat_inverse_transform_points(p, q, X):
    X = np.array(X, dtype=float)
    return np.vstack([
        quat_rotate_vector(quat_conjugate(q), x - p)
        for x in X
    ])

def quat_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)

def quat_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3,)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / angle
    return quat_from_axis_angle(axis, angle)

class SimulateGripper:

    def __init__(self, cloth: Cloth, box_size=np.array([0.03, 0.06, 0.02], dtype=float)):

        self.cloth: Cloth = cloth

        self.controlled = [] # grasped nodes
        self.local_points = [] # grasped points in local gripper frame

        # pose
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.p = np.array([0.0, 0.0, 0.0], dtype=float)

        ## Parallelopiped
        # dimensions of grasp region in gripper local frame
        # x = jaw opening direction
        # y = finger thickness direction
        # z = finger length / approach direction
        self.box_size = box_size
        
        self.is_open = True

        # So that the gripper is only shown when it grasps something
        self.origin_history = [np.array([0.0, 0.0, 0.0], dtype=float)]
        self.R_history = [np.eye(3)]
        self.visible_history = [True]
        self.grasp_history = [False]

        # squeeze attributes
        self.local_points_rest = np.zeros((0, 3), dtype=float)
        self.local_points_goal = np.zeros((0, 3), dtype=float)
        self.squeeze_alpha = 1.0 # changed to 0
        self.squeeze_alpha_step = 0.1   # smaller = safer
        self.squeeze_amount = 0.50

    def set_pose(self, q=None, p=None):
        if q is None:
            q = [1.0, 0.0, 0.0, 0.0]
        if p is None:
            p = [0.0, 0.0, 0.0]

        self.q = quat_normalize(q)
        self.p = np.array(p, dtype=float).reshape(3,)

# quad center in grasp box OR quad center near to grasp box origin

    def find_nodes_in_vicinity(self, smooth=2, box=None, center_local=None):
        if box is None:
            box = self.box_size
        if center_local is None:
            center_local = np.zeros(3, dtype=float)

        box = np.asarray(box, dtype=float).reshape(3,)
        center_local = np.asarray(center_local, dtype=float).reshape(3,)
        half = 0.5 * box

    ## nodes: Take the smoothed ones rather than the actual ones
        phi_mat = self.cloth.positions
        phi_all = self.cloth.Am @ phi_mat
        for _ in range(smooth):
            phi_all = self.cloth.S @ phi_all

        n_nodes = self.cloth.positions.shape[0]
        Xw_nodes = phi_all[:n_nodes]

        # Xw_nodes = self.cloth.positions
        Xl_nodes = quat_inverse_transform_points(self.p, self.q, Xw_nodes)
        Xc_nodes = Xl_nodes - center_local.reshape(1, 3)

        # inside nodes is an array of booleans [False, True, False, ...]
        inside_nodes = (
            (np.abs(Xc_nodes[:, 0]) <= half[0]) &
            (np.abs(Xc_nodes[:, 1]) <= half[1]) &
            (np.abs(Xc_nodes[:, 2]) <= half[2])
        )

        support = set(np.where(inside_nodes)[0].tolist())

    ## If quad center lies in the grasp box, select all four nodes

        face_nodes = self.cloth.faces
        # Xw_face_centers = phi_all[n_nodes:]
        Xw_face_centers = Xw_nodes[face_nodes].mean(axis=1)

        # quad centers in local frame
        Xl_face_centers = quat_inverse_transform_points(self.p, self.q, Xw_face_centers)
        Xc_face_centers = Xl_face_centers - center_local.reshape(1, 3)

        inside_face_nodes = (
                (np.abs(Xc_face_centers[:, 0]) <= half[0]) &
                (np.abs(Xc_face_centers[:, 1]) <= half[1]) &
                (np.abs(Xc_face_centers[:, 2]) <= half[2])
            )
        
        support_face_nodes = np.where(inside_face_nodes)[0]

        if support_face_nodes.size > 0:
            # add individual nodes as int
            support.update(
                int(i) for i in np.unique(face_nodes[support_face_nodes].reshape(-1))
            )

    ## If center of an edge lies in the grasp box, select the nodes of that edge
        
        edge_nodes = self.cloth.edges_matrix
        Xw_edge_centers = Xw_nodes[edge_nodes].mean(axis=1)

         # quad centers in local frame
        Xl_edge_centers = quat_inverse_transform_points(self.p, self.q, Xw_edge_centers)
        Xc_edge_centers = Xl_edge_centers - center_local.reshape(1, 3)

        inside_edge_nodes = (
                (np.abs(Xc_edge_centers[:, 0]) <= half[0]) &
                (np.abs(Xc_edge_centers[:, 1]) <= half[1]) &
                (np.abs(Xc_edge_centers[:, 2]) <= half[2])
            )
        
        support_edge_nodes = np.where(inside_edge_nodes)[0]
        if support_edge_nodes.size > 0:
            # add individual nodes as int
            support.update(
                int(i) for i in np.unique(edge_nodes[support_edge_nodes].reshape(-1))
            )


        return sorted(support)

    
    def set_open(self, is_open, smooth, box=None, center_local=None):

        is_open = bool(is_open)

        was_open = self.is_open
        self.is_open = is_open

        # open -> closed : attempt grasp
        if was_open and (not self.is_open):
            inds = self.find_nodes_in_vicinity(box=box, smooth=smooth, center_local=center_local)            
            # print(f'grasped nodes: {inds}') # only print when changing from open to closed
            if len(inds) > 0:
                self.controlled = inds
                Xw = self.cloth.positions[self.controlled].copy()
                Xl = quat_inverse_transform_points(self.p, self.q, Xw)

                # store unsqueezed grasp points
                self.local_points_rest = Xl.copy()

                # make a safe squeezed target for ALL grasped nodes
                Xl_goal = Xl.copy()

                dx = center_local[0] - Xl_goal[:, 0]
                dx *= self.squeeze_amount      # move only 50% toward center in gripper x
                # dx = np.clip(dx, -0.002, 0.002)  # cap to 2 mm per node
                Xl_goal[:, 0] += dx

                Xl_goal[:, 2] += 0.0002       # tiny lift to reduce sudden jump

                self.local_points_goal = Xl_goal
                self.local_points = self.local_points_rest.copy()
                self.squeeze_alpha = 0.0

            else:
                self.controlled = []
                self.local_points = np.zeros((0, 3))
                self.local_points_rest = np.zeros((0, 3))
                self.local_points_goal = np.zeros((0, 3))
                self.squeeze_alpha = 0.0 # changed to 0

        # closed -> open : release
        elif (not was_open) and self.is_open:
            self.controlled = []
            self.local_points = np.zeros((0, 3))
            self.local_points_rest = np.zeros((0, 3))
            self.local_points_goal = np.zeros((0, 3))
            self.squeeze_alpha = 1.0

    def record_history(self):
        R = quat_to_rotmat(self.q)
        grasp_now = (not self.is_open) and (len(self.controlled) > 0)

        self.origin_history.append(self.p.copy())
        self.R_history.append(R.copy())
        self.visible_history.append(True)
        self.grasp_history.append(grasp_now)
                
    def step(self):
        # step is called on every callback frame, hence the target positions for the grasped nodes
        # are sent to the solver every frame, not once. Thus, gradual squeezing can be done
        # to avoid sudden jumping: from local_points_rest to local_points_goal.
        if (not self.is_open) and len(self.controlled) > 0:
            if self.squeeze_alpha < 1.0:
                self.squeeze_alpha = min(1.0, self.squeeze_alpha + self.squeeze_alpha_step)
                # a = 0.15, next step, a = min(1.0, 0.15 + 0.10) = 0.25, next step, a = 0.35, ...
                a = self.squeeze_alpha
                # changing self.local_points every frame, even though self.controlled does not change.
                self.local_points = (1.0 - a) * self.local_points_rest + a * self.local_points_goal

            u = quat_transform_points(self.p, self.q, self.local_points)
            self.cloth.simulate(u=u, control=self.controlled)
        else:
            self.cloth.simulate(u=np.zeros((0, 3)), control=[])

        self.record_history()
        