import ctypes
import numpy as np

from implementation.Cloth_speed import Cloth
from implementation.Gripper_for_Unity import SimulateGripper

class UnityClothWithGripper:
    """
    Unity-Python wrapper.

    Unity sends to Python:
      - gripper position
      - gripper quaternion
      - gripper box size
      - open/closed state

    Python decides:
      - which cloth nodes are grasped
      - squeeze motion
      - controlled node positions
      - cloth simulation
    """

    def __init__(self, verts, faces):
        self.cloth = Cloth(np.array(verts, dtype=float), np.array(faces, dtype=int))
        # self.gripper = SimulateGripper(self.cloth)
        self.grippers = {}

        self.grasp_smooth = 2
        self.center_local = np.zeros(3, dtype=float)

    def setSimulatorParameters(self, **kwargs):
        self.cloth.setSimulatorParameters(**kwargs)

    def _float_ptr_to_np(self, ptr, n):
        """
        converting a pointer to np array
        To pin an array to an address in C#, the pointer
        pointing to that address is also sent to Python
        but we only need the corresponding array
        """
        arr_type = ctypes.c_float * n
        return np.ctypeslib.as_array(arr_type.from_address(int(ptr))).copy()
    
    def _get_gripper(self, gripper_id):
        if not hasattr(self, "grippers"):
            self.grippers = {}

        gripper_id = int(gripper_id)

        if gripper_id not in self.grippers:
            self.grippers[gripper_id] = SimulateGripper(self.cloth)

        return self.grippers[gripper_id]

    def simulate_gripper(self, p_ptr, q_ptr, box_ptr, closed):
        self.simulate_gripper_id(0, p_ptr, q_ptr, box_ptr, closed)


    def simulate_gripper_id(self, gripper_id, p_ptr, q_ptr, box_ptr, closed):
        p = self._float_ptr_to_np(p_ptr, 3)
        q = self._float_ptr_to_np(q_ptr, 4)
        box = self._float_ptr_to_np(box_ptr, 3)

        g = self._get_gripper(gripper_id)

        g.box_size = box
        g.set_pose(q=q, p=p)

        g.set_open(
            is_open=(not bool(closed)),
            smooth=self.grasp_smooth,
            box=box,
            center_local=self.center_local,
        )

        g.step(self.grippers.values())
    
    def get_grasped_node_ids(self):
        ids = []

        for g in self.grippers.values():
            ids.extend(int(i) for i in g.controlled)

        return sorted(set(ids))

    def getPositionsUnity(self, smooth):
        phi_all = self.cloth.Am @ self.cloth.positions
        for _ in range(smooth):
            phi_all = self.cloth.S @ phi_all
        return phi_all.tolist()
    
    def getPhysicalPositionsUnity(self):
        return self.cloth.positions.tolist()
    
    def get_raw_velocities(self):
        return self.cloth.velocities.tolist()