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
        self.gripper = SimulateGripper(self.cloth)

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

    def simulate_gripper(self, p_ptr, q_ptr, box_ptr, closed):
        """
        Called from Unity every frame.

        p_ptr   : float[3], Python frame position
        q_ptr   : float[4], Python quaternion [w, x, y, z]
        box_ptr : float[3], Python local box size
        closed  : bool
        """

        p = self._float_ptr_to_np(p_ptr, 3)
        q = self._float_ptr_to_np(q_ptr, 4)
        box = self._float_ptr_to_np(box_ptr, 3)

        self.gripper.box_size = box
        self.gripper.set_pose(q=q, p=p)

        # open -> closed triggers Python-side node detection
        # closed -> open releases nodes
        self.gripper.set_open(
            is_open=(not bool(closed)),
            smooth=self.grasp_smooth,
            box=box,
            center_local=self.center_local,
        )

        # Python applies the controlled node positions to cloth.simulate(...)
        self.gripper.step()

    def get_grasped_node_ids(self):
        return [int(i) for i in self.gripper.controlled]

    def getPositionsUnity(self, smooth):
        phi_all = self.cloth.Am @ self.cloth.positions
        for _ in range(smooth):
            phi_all = self.cloth.S @ phi_all
        return phi_all.tolist()
    
    def getPhysicalPositionsUnity(self):
        return self.cloth.positions.tolist()