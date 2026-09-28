# Replay a Unity cloth simulation with TWO grippers from replay_cloth_2.bin.
#
# Expected binary format:
#   version = 2
#   gripper index 0 = LEFT
#   gripper index 1 = RIGHT
#   pose order = [px, py, pz, qw, qx, qy, qz]
#
# This follows the same replay structure as replay_gripper_box_Unity_2grippers.py:
# - simulate the cloth again in Python
# - overlay the cloth recorded in Unity
# - replay the two exported graspFrame poses
# - replay both jaw open/close states
# - use TWO SimulateGripper objects, but advance the cloth only ONCE per frame

import os
import sys
import time
from pathlib import Path

import numpy as np
import polyscope as ps


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))

CLOTHILDE_ROOT = os.path.join(parent_dir, "python_code")
sys.path.insert(0, CLOTHILDE_ROOT)

EXPORT_DIR = os.path.join(CLOTHILDE_ROOT, "exported_data3_binary")

from implementation.Cloth_speed import Cloth
from implementation.Gripper import (
    SimulateGripper,
    quat_transform_points,
    quat_normalize,
)


# ---------------------------------------------------------------------------
# Replay settings
# ---------------------------------------------------------------------------

smooth = 2

show_graspbox = True

# IMPORTANT:
# This must match the graspBoxSize used in Unity.
# Your earlier Python replay used 40 mm.
# If the Unity Inspector uses 0.06, 0.06, 0.06 instead, change this to [0.06, 0.06, 0.06].
grasp_box = np.array([0.04, 0.04, 0.04], dtype=float)

# The exported pose is graspFrame itself, so the box is centered at the frame origin.
tip_center_local = np.zeros(3, dtype=float)

repeat = False


# ---------------------------------------------------------------------------
# Locate binary
# ---------------------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

candidate_paths = [
    script_dir / "replay_mantel.bin",
    Path(EXPORT_DIR) / "replay_mantel.bin",
]

BIN_PATH = None
for candidate in candidate_paths:
    if candidate.exists():
        BIN_PATH = candidate
        break

if BIN_PATH is None:
    raise FileNotFoundError(
        "Could not find replay_cloth_2.bin.\n"
        "Put it next to this script or in:\n"
        f"  {EXPORT_DIR}"
    )


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "dt", "tol", "sub_steps", "rho", "delta", "alpha",
    "kappa", "kappa_bnd", "str", "shr", "slf",
    "mu_f", "mu_s", "thck",
]


def read_unity_binary(path):
    with open(path, "rb") as f:
        magic = f.read(8).decode("ascii")

        if magic != "CLTHSIM1":
            raise ValueError(f"Wrong file format: {magic!r}")

        version_arr = np.fromfile(f, dtype="<i4", count=1)
        if version_arr.size != 1:
            raise ValueError("Could not read file version.")

        version = int(version_arr[0])

        if version != 2:
            raise ValueError(
                f"This replay expects the two-gripper binary format (version 2), "
                f"but this file is version {version}."
            )

        header = np.fromfile(f, dtype="<i4", count=5)
        if header.size != 5:
            raise ValueError("Incomplete version-2 header.")

        n_frames, n_vertices, n_faces, n_params, n_grippers = map(int, header)

        if n_grippers != 2:
            raise ValueError(
                f"This replay expects exactly 2 grippers, but the file reports "
                f"{n_grippers}."
            )

        mesh_vertices = np.fromfile(
            f, dtype="<f8", count=n_vertices * 3
        ).reshape(n_vertices, 3)

        mesh_faces = np.fromfile(
            f, dtype="<i4", count=n_faces * 4
        ).reshape(n_faces, 4)

        cloth_frames = np.fromfile(
            f, dtype="<f8", count=n_frames * n_vertices * 3
        ).reshape(n_frames, n_vertices, 3)

        gripper_poses = np.fromfile(
            f, dtype="<f4", count=n_frames * n_grippers * 7
        ).reshape(n_frames, n_grippers, 7)

        jaw_status = np.fromfile(
            f, dtype="<i4", count=n_frames * n_grippers
        ).reshape(n_frames, n_grippers)

        times = np.fromfile(
            f, dtype="<f4", count=n_frames
        )

        sim_params_arr = np.fromfile(
            f, dtype="<f4", count=n_params
        )

        remaining = f.read()
        if remaining:
            raise ValueError(
                f"Binary layout mismatch: {len(remaining)} unread bytes remain."
            )

    if len(sim_params_arr) != n_params:
        raise ValueError(
            f"Expected {n_params} simulator parameters, "
            f"read {len(sim_params_arr)}."
        )

    params = dict(zip(PARAM_NAMES[:n_params], sim_params_arr))
    if "sub_steps" in params:
        params["sub_steps"] = int(params["sub_steps"])

    print("Loaded:", path)
    print(
        f"version={version}, frames={n_frames}, vertices={n_vertices}, "
        f"faces={n_faces}, grippers={n_grippers}, params={n_params}"
    )

    return (
        mesh_vertices,
        mesh_faces,
        cloth_frames,
        gripper_poses,
        jaw_status,
        times,
        params,
    )


# ---------------------------------------------------------------------------
# Grasp-box geometry
# ---------------------------------------------------------------------------

box_faces = np.array(
    [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=int,
)


def get_box_vertices_world_offset(p, q, box_size, center_local):
    hx, hy, hz = 0.5 * np.asarray(box_size, dtype=float)
    c = np.asarray(center_local, dtype=float).reshape(3,)

    vertices_local = np.array(
        [
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ],
        dtype=float,
    )

    vertices_local += c.reshape(1, 3)

    return quat_transform_points(
        np.asarray(p, dtype=float),
        quat_normalize(q),
        vertices_local,
    )


# ---------------------------------------------------------------------------
# Load recording
# ---------------------------------------------------------------------------

(
    X0,
    T0,
    frames,
    gripper_poses,
    jaw_status,
    times,
    params,
) = read_unity_binary(BIN_PATH)

n_frames = len(frames)

print("Mesh:", X0.shape)
print("Faces:", T0.shape)
print("Recorded cloth:", frames.shape)
print("Gripper poses:", gripper_poses.shape)
print("Jaw status:", jaw_status.shape)
print("Times:", times.shape)
print("Initial mesh/frame max error:", np.max(np.abs(frames[0] - X0)))


# ---------------------------------------------------------------------------
# Split the two grippers
# ---------------------------------------------------------------------------

# Binary convention:
#   gripper 0 = LEFT
#   gripper 1 = RIGHT

p_left = gripper_poses[:, 0, 0:3]
q_left = gripper_poses[:, 0, 3:7]

p_right = gripper_poses[:, 1, 0:3]
q_right = gripper_poses[:, 1, 3:7]

jaw_left = jaw_status[:, 0].astype(bool)
jaw_right = jaw_status[:, 1].astype(bool)


# ---------------------------------------------------------------------------
# Python cloth simulation
# ---------------------------------------------------------------------------

X_reset = X0.copy()

cloth = Cloth(X0.copy(), T0.copy())

# Use the simulator parameters stored by Unity.
cloth.setSimulatorParameters(**params)
cloth.preparePolyscope()


# ---------------------------------------------------------------------------
# Recorded Unity cloth overlay
# ---------------------------------------------------------------------------

cloth_unity = Cloth(X0.copy(), T0.copy())
cloth_unity.positions = X0.copy()
cloth_unity.label = "recorded_cloth"

phi2_0 = cloth_unity.Am @ cloth_unity.positions
for _ in range(smooth):
    phi2_0 = cloth_unity.S @ phi2_0

rec_mesh = ps.register_surface_mesh(
    cloth_unity.label,
    phi2_0,
    cloth_unity.triangles,
    color=[0.55, 0.90, 0.55],
    transparency=0.45,
    smooth_shade=True,
    edge_width=0,
)

rec_pc = ps.register_point_cloud(
    cloth_unity.label + "_nodes",
    cloth_unity.positions,
    enabled=False,
)


# ---------------------------------------------------------------------------
# Two Python grippers
# ---------------------------------------------------------------------------

grip_left = SimulateGripper(cloth, box_size=grasp_box)
grip_right = SimulateGripper(cloth, box_size=grasp_box)

# Initialize their poses before the first callback.
grip_left.set_pose(q_left[0], p_left[0])
grip_right.set_pose(q_right[0], p_right[0])


# ---------------------------------------------------------------------------
# Grasp-box visualization
# ---------------------------------------------------------------------------

if show_graspbox:
    V_left_0 = get_box_vertices_world_offset(
        p_left[0],
        q_left[0],
        grasp_box,
        tip_center_local,
    )

    V_right_0 = get_box_vertices_world_offset(
        p_right[0],
        q_right[0],
        grasp_box,
        tip_center_local,
    )

    ps.register_surface_mesh(
        "grasp_box_left",
        V_left_0,
        box_faces,
        color=[0.0, 1.0, 0.0],
        transparency=0.55,
        material="wax",
    )

    ps.register_surface_mesh(
        "grasp_box_right",
        V_right_0,
        box_faces,
        color=[0.0, 0.6, 1.0],
        transparency=0.55,
        material="wax",
    )


# ---------------------------------------------------------------------------
# Playback state
# ---------------------------------------------------------------------------

# Use recorded Unity frame timing for visualization.
if len(times) > 1:
    positive_diffs = np.diff(times)
    positive_diffs = positive_diffs[positive_diffs > 0]

    if len(positive_diffs) > 0:
        default_frame_dt = float(np.median(positive_diffs))
    else:
        default_frame_dt = 1.0 / 60.0
else:
    default_frame_dt = 1.0 / 60.0


state = {
    "k": 0,
    "last_time": time.time(),
    "paused": False,
}


# ---------------------------------------------------------------------------
# Scene update
# ---------------------------------------------------------------------------

def update_scene(k):
    # -------------------------------------------------------
    # Python-simulated cloth
    # -------------------------------------------------------

    phi_mat = cloth.positions

    phi_all = cloth.Am @ phi_mat
    for _ in range(smooth):
        phi_all = cloth.S @ phi_all

    # The smoothed display should not move controlled nodes away
    # from their exact grasp constraints.
    controlled = set(grip_left.controlled) | set(grip_right.controlled)

    if controlled:
        ctrl = np.asarray(sorted(controlled), dtype=int)
        phi_all[ctrl] = phi_mat[ctrl]

    ps.get_surface_mesh(cloth.label).update_vertex_positions(phi_all)
    ps.get_point_cloud(cloth.label).update_point_positions(phi_mat)

    # -------------------------------------------------------
    # Unity-recorded cloth
    # -------------------------------------------------------

    cloth_unity.positions = frames[k]

    phi2_all = cloth_unity.Am @ cloth_unity.positions
    for _ in range(smooth):
        phi2_all = cloth_unity.S @ phi2_all

    rec_mesh.update_vertex_positions(phi2_all)
    rec_pc.update_point_positions(cloth_unity.positions)


# ---------------------------------------------------------------------------
# Replay callback
# ---------------------------------------------------------------------------

def callback():
    if state["paused"]:
        return

    k = state["k"]

    # Respect the recorded inter-frame timing when possible.
    if k == 0:
        frame_dt = default_frame_dt
    else:
        frame_dt = float(times[k] - times[k - 1])
        if frame_dt <= 0 or not np.isfinite(frame_dt):
            frame_dt = default_frame_dt

    now = time.time()

    if now - state["last_time"] < frame_dt:
        return

    state["last_time"] = now

    left_open = bool(jaw_left[k])
    right_open = bool(jaw_right[k])

    # -------------------------------------------------------
    # Update visual grasp boxes
    # -------------------------------------------------------

    if show_graspbox:
        V_left = get_box_vertices_world_offset(
            p_left[k],
            q_left[k],
            grasp_box,
            tip_center_local,
        )

        V_right = get_box_vertices_world_offset(
            p_right[k],
            q_right[k],
            grasp_box,
            tip_center_local,
        )

        left_box = ps.get_surface_mesh("grasp_box_left")
        right_box = ps.get_surface_mesh("grasp_box_right")

        left_box.update_vertex_positions(V_left)
        right_box.update_vertex_positions(V_right)

        # Open = distinct left/right color; closed = red.
        if left_open:
            left_box.set_color([0.0, 1.0, 0.0])
        else:
            left_box.set_color([1.0, 0.0, 0.0])

        if right_open:
            right_box.set_color([0.0, 0.6, 1.0])
        else:
            right_box.set_color([1.0, 0.0, 0.0])

    # -------------------------------------------------------
    # Update BOTH gripper poses
    # -------------------------------------------------------

    grip_left.set_pose(q_left[k], p_left[k])
    grip_right.set_pose(q_right[k], p_right[k])

    # Detect/release grasped nodes independently.
    grip_left.set_open(
        is_open=left_open,
        smooth=smooth,
        box=grasp_box,
        center_local=tip_center_local,
        squeeze_enabled=True,
    )

    grip_right.set_open(
        is_open=right_open,
        smooth=smooth,
        box=grasp_box,
        center_local=tip_center_local,
        squeeze_enabled=True,
    )

    # IMPORTANT:
    # Advance physics ONCE with both grippers.
    #
    # Do not call grip_left.step() and grip_right.step() separately,
    # because that would advance the cloth twice for one Unity frame.
    grip_left.step(grippers=[grip_left, grip_right])

    update_scene(k)

    # Optional terminal diagnostics when grasp sets change.
    if k == 0 or jaw_left[k] != jaw_left[k - 1] or jaw_right[k] != jaw_right[k - 1]:
        print(
            f"frame={k:4d}  "
            f"left_open={left_open} controlled={list(grip_left.controlled)}  "
            f"right_open={right_open} controlled={list(grip_right.controlled)}"
        )

    # -------------------------------------------------------
    # Advance frame
    # -------------------------------------------------------

    state["k"] += 1

    if state["k"] >= n_frames:
        if repeat:
            state["k"] = 0
            state["last_time"] = time.time()

            cloth.positions = X_reset.copy()
            cloth_unity.positions = X_reset.copy()

            print("Replay restarted.")
        else:
            state["k"] = n_frames - 1
            state["paused"] = True
            print("Reached final frame. Simulation stopped.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# Show the initial recorded/simulated state immediately.
update_scene(0)

ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
