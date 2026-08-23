"""
FILE 1: teleop/env.py  —  FrankaMujocoEnv

Wraps MuJoCo's model/data behind a clean interface, so nothing else in the
project touches MuJoCo internals directly.
"""

import numpy as np
import mujoco


# Panda home pose (7 arm joints). A slightly-bent "ready" configuration —
# the all-zeros pose is singular and makes IK misbehave on the first step.
PANDA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


class FrankaMujocoEnv:

    def __init__(
        self,
        xml_path: str,
        camera_names: list,
        render_size=(224, 224),
        control_hz: int = 15,
        table_top_z: float = 0.0,
        lift_threshold: float = 0.05,
    ):
        self.xml_path = xml_path
        self.camera_names = list(camera_names)
        self.render_size = render_size
        self.control_hz = control_hz
        self.table_top_z = table_top_z
        self.lift_threshold = lift_threshold

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- names (mujoco_menagerie Panda conventions) --------------------
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_actuator_names = [f"actuator{i}" for i in range(1, 8)]
        self.gripper_actuator_name = "actuator8"
        self.ee_site_name = "ee_site"
        self.object_body_name = "target_object"
        self.object_joint_name = "target_object_freejoint"

        # --- resolve names -> ids ONCE -------------------------------------
        self._arm_joint_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names]
        )
        # joint id != qpos index != dof index. Look each up properly.
        self._arm_qpos_ids = np.array(
            [self.model.jnt_qposadr[j] for j in self._arm_joint_ids]
        )
        self._arm_dof_ids = np.array(
            [self.model.jnt_dofadr[j] for j in self._arm_joint_ids]
        )
        self._arm_actuator_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.arm_actuator_names]
        )
        self._gripper_actuator_id = self._name2id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator_name
        )
        self._ee_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self._object_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, self.object_body_name)
        self._object_qpos_adr = self.model.jnt_qposadr[
            self._name2id(mujoco.mjtObj.mjOBJ_JOINT, self.object_joint_name)
        ]

        # Read the gripper's ctrlrange from the model instead of hardcoding it.
        self.gripper_ctrlrange = self.model.actuator_ctrlrange[self._gripper_actuator_id].copy()

        # --- physics substeps per control tick ------------------------------
        self.n_substeps = max(1, int(round((1.0 / control_hz) / self.model.opt.timestep)))

        # --- renderers (expensive; create once) -----------------------------
        h, w = render_size
        self._renderers = {
            name: mujoco.Renderer(self.model, height=h, width=w) for name in self.camera_names
        }

        self._rng = np.random.default_rng()

    # ----------------------------------------------------------------------
    def _name2id(self, objtype, name):
        oid = mujoco.mj_name2id(self.model, objtype, name)
        if oid < 0:
            raise ValueError(
                f"'{name}' not found in the model. Check your XML — names must match exactly."
            )
        return oid

    # ----------------------------------------------------------------------
    def reset(self, randomize_object: bool = True):
        mujoco.mj_resetData(self.model, self.data)

        # arm to home pose (both state and control target)
        self.data.qpos[self._arm_qpos_ids] = PANDA_HOME_QPOS
        self.data.ctrl[self._arm_actuator_ids] = PANDA_HOME_QPOS
        # gripper open
        self.data.ctrl[self._gripper_actuator_id] = self.gripper_ctrlrange[1]

        # object freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
        a = self._object_qpos_adr
        if randomize_object:
            x = self._rng.uniform(0.35, 0.55)
            y = self._rng.uniform(-0.15, 0.15)
        else:
            x, y = 0.45, 0.0
        self.data.qpos[a : a + 3] = [x, y, self.table_top_z + 0.02]
        self.data.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        # let the object settle onto the table before the first observation
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------------------
    def step(self, q_target: np.ndarray, gripper_ctrl: float):
        self.data.ctrl[self._arm_actuator_ids] = q_target
        self.data.ctrl[self._gripper_actuator_id] = gripper_ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------------------
    def get_ee_pose(self):
        """(position (3,), quaternion (4,) in MuJoCo [w,x,y,z] order)."""
        pos = self.data.site_xpos[self._ee_site_id].copy()
        mat = self.data.site_xmat[self._ee_site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    # ----------------------------------------------------------------------
    def get_jacobian(self) -> np.ndarray:
        """6x7 Jacobian of ee_site w.r.t. the arm joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        J = np.vstack([jacp, jacr])          # 6 x nv (nv includes object DOFs)
        return J[:, self._arm_dof_ids]       # 6 x 7 (arm columns only)

    # ----------------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_ids].copy()

    def get_joint_limits(self) -> np.ndarray:
        return self.model.jnt_range[self._arm_joint_ids].copy()

    def get_gripper_width(self) -> float:
        """Normalized 0 (closed) .. 1 (open), derived from the actuator ctrl."""
        lo, hi = self.gripper_ctrlrange
        if hi - lo < 1e-9:
            return 0.0
        return float((self.data.ctrl[self._gripper_actuator_id] - lo) / (hi - lo))

    # ----------------------------------------------------------------------
    def render_all_cameras(self) -> dict:
        images = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render().copy()
        return images

    # ----------------------------------------------------------------------
    def get_object_position(self) -> np.ndarray:
        return self.data.xpos[self._object_body_id].copy()

    def is_grasped(self) -> bool:
        """Crude success check: object lifted clear of the table."""
        return bool(self.get_object_position()[2] > self.table_top_z + self.lift_threshold)

    # ----------------------------------------------------------------------
    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass