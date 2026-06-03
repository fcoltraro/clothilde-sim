#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

# ============================================================
# EDIT ONLY THESE PATHS
# ============================================================

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))

# This is the folder containing implementation/Cloth.py and implementation/Gripper.py
CLOTHILDE_ROOT = parent_dir + "/python_code"
sys.path.append(CLOTHILDE_ROOT)

# Folder containing:
#   mesh_vertices.csv
#   mesh_faces.csv
#   gripper_poses.csv
EXPORT_DIR = CLOTHILDE_ROOT + "/exported_data"

# Simulation parameters
DT = 1.0 / 60.0
SUB_STEPS = 9
TOL = 0.0075
SMOOTH = 2
SHOW_MOVIE = True
MOVIE_SPEED = 1

# ============================================================

sys.path.insert(0, str(CLOTHILDE_ROOT))

from implementation.Cloth import Cloth
from implementation.Gripper import SimulateGripper, quat_transform_points

def read_mesh(export_dir):
    vertices_path = os.path.join(export_dir, "mesh_vertices.csv")
    faces_path = os.path.join(export_dir, "mesh_faces.csv")

    vdf = pd.read_csv(vertices_path).sort_values("node_index")
    fdf = pd.read_csv(faces_path).sort_values("face_id")

    X = vdf[["x", "y", "z"]].to_numpy(dtype=float)
    T = fdf[["n0", "n1", "n2", "n3"]].to_numpy(dtype=int)

    return X, T

def read_gripper_poses(export_dir):
    poses_path = os.path.join(export_dir, "gripper_poses.csv")
    poses = pd.read_csv(poses_path)

    poses["frame"] = poses["frame"].astype(int)
    poses = poses.sort_values(["frame", "hand"]).reset_index(drop=True)

    return poses


def row_to_pose(row):
    p = row[["p_x", "p_y", "p_z"]].to_numpy(dtype=float)

    # Already exported in clothilde-sim convention:
    # q = [qw, qx, qy, qz]
    q = row[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)

    return p, q


def override_squeeze_goal_to_match_unity(
    gripper,
    center_local,
    squeeze_amount,
    enable_squeeze,
):
    """
    Match GripperVR.cs squeeze:

        goalLocal.x += squeezeAmount * (centerLocal.x - goalLocal.x)
    """

    if len(gripper.controlled) == 0:
        return

    Xl_goal = gripper.local_points_rest.copy()

    if enable_squeeze:
        dx = center_local[0] - Xl_goal[:, 0]
        Xl_goal[:, 0] += squeeze_amount * dx

    gripper.local_points_goal = Xl_goal
    gripper.local_points = gripper.local_points_rest.copy()
    gripper.squeeze_alpha = 0.0

def collect_gripper_target(gripper):
    """
    Same logic as SimulateGripper.step(), but without directly calling
    cloth.simulate(). This allows multiple grippers in one simulation step.
    """

    if (not gripper.is_open) and len(gripper.controlled) > 0:
        if gripper.squeeze_alpha < 1.0:
            gripper.squeeze_alpha = min(
                1.0,
                gripper.squeeze_alpha + gripper.squeeze_alpha_step,
            )

            a = gripper.squeeze_alpha

            gripper.local_points = (
                (1.0 - a) * gripper.local_points_rest
                + a * gripper.local_points_goal
            )

        u = quat_transform_points(
            gripper.p,
            gripper.q,
            gripper.local_points,
        )

        return list(gripper.controlled), np.asarray(u, dtype=float)

    return [], np.zeros((0, 3), dtype=float)


def combine_controls(control_blocks, u_blocks):
    if len(control_blocks) == 0:
        return [], np.zeros((0, 3), dtype=float)

    all_control = []
    all_u = []

    for control, u in zip(control_blocks, u_blocks):
        all_control.extend(control)
        all_u.append(u)

    all_u = np.vstack(all_u)

    # If two grippers control the same node, keep the last command.
    dedup = {}
    for node, pos in zip(all_control, all_u):
        dedup[int(node)] = pos

    control = list(dedup.keys())
    u = np.vstack([dedup[i] for i in control])

    return control, u


def main():
    X, T = read_mesh(EXPORT_DIR)
    poses = read_gripper_poses(EXPORT_DIR)

    cloth = Cloth(X, T)
    cloth.setSimulatorParameters(
        dt=DT,
        sub_steps=SUB_STEPS,
        tol=TOL,
    )

    grippers = {}

    for frame, group in poses.groupby("frame", sort=True):
        control_blocks = []
        u_blocks = []

        for _, row in group.iterrows():
            hand = str(row["hand"])

            box = row[["box_x", "box_y", "box_z"]].to_numpy(dtype=float)
            center_local = row[
                ["center_x", "center_y", "center_z"]
            ].to_numpy(dtype=float)

            if hand not in grippers:
                grippers[hand] = SimulateGripper(
                    cloth,
                    box_size=box,
                )

            gripper = grippers[hand]

            p, q = row_to_pose(row)
            gripper.set_pose(q=q, p=p)

            gripper.squeeze_alpha_step = float(row["squeeze_alpha_step"])

            is_open = bool(int(row["is_open"]))
            close_event = bool(int(row["close_event"]))

            gripper.set_open(
                is_open=is_open,
                smooth=SMOOTH,
                box=box,
                center_local=center_local,
            )

            if close_event:
                enable_squeeze = bool(int(row["enable_squeeze"]))
                squeeze_amount = float(row["squeeze_amount"])

                override_squeeze_goal_to_match_unity(
                    gripper,
                    center_local=center_local,
                    squeeze_amount=squeeze_amount,
                    enable_squeeze=enable_squeeze,
                )

            control, u = collect_gripper_target(gripper)

            if len(control) > 0:
                control_blocks.append(control)
                u_blocks.append(u)

        control, u = combine_controls(control_blocks, u_blocks)

        cloth.simulate(u=u, control=control)

        for gripper in grippers.values():
            gripper.record_history()

    print(f"Replayed {poses['frame'].nunique()} frames.")
    print(f"Cloth history frames: {len(cloth.history_pos)}")

    if len(cloth.history_pos) > 1:
        avg_iters = cloth.total_iters / (len(cloth.history_pos) - 1)
        print(f"Average solver iterations: {avg_iters:.3f}")

    if SHOW_MOVIE:
        cloth.makeMovie(
            speed=MOVIE_SPEED,
            repeat=True,
            smooth=SMOOTH,
        )


if __name__ == "__main__":
    main()