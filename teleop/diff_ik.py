"""
FILE 3: teleop/diff_ik.py  —  DiffIKSolver

Differential inverse kinematics via damped least squares.
Pure numpy: no MuJoCo dependency, so it's testable standalone.
"""

import numpy as np


class DiffIKSolver:

    def __init__(self, damping: float = 0.05, max_dq: float = 0.2):
        """
        damping : keeps the solve stable near singularities (larger = smoother
                  but sloppier tracking; smaller = accurate but can spike).
        max_dq  : hard safety clamp on per-step joint motion (radians). Stops a
                  bad solve from flinging the arm across the scene.
        """
        self.damping = damping
        self.max_dq = max_dq

    # ----------------------------------------------------------------------
    def solve(
        self,
        current_ee_pos: np.ndarray,
        current_ee_quat: np.ndarray,
        target_ee_pos: np.ndarray,
        target_ee_quat: np.ndarray,
        jacobian: np.ndarray,
        current_joint_positions: np.ndarray,
        joint_limits: np.ndarray = None,
    ) -> np.ndarray:
        pos_err = np.asarray(target_ee_pos) - np.asarray(current_ee_pos)
        ori_err = self._quat_error(target_ee_quat, current_ee_quat)
        err_6d = np.concatenate([pos_err, ori_err])

        J = np.asarray(jacobian)
        lambda_sq = self.damping ** 2
        # damped least squares: dq = J^T (J J^T + l^2 I)^-1 e
        # np.linalg.solve is preferred over explicit inv(): stabler and faster.
        dq = J.T @ np.linalg.solve(J @ J.T + lambda_sq * np.eye(6), err_6d)

        # safety clamp on step magnitude
        dq_norm = np.linalg.norm(dq)
        if dq_norm > self.max_dq:
            dq = dq * (self.max_dq / dq_norm)

        q_target = np.asarray(current_joint_positions) + dq

        if joint_limits is not None:
            joint_limits = np.asarray(joint_limits)
            q_target = np.clip(q_target, joint_limits[:, 0], joint_limits[:, 1])

        return q_target

    # ----------------------------------------------------------------------
    def _quat_error(self, target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        """Small-angle rotational error as a (3,) axis-angle vector.

        Quaternions are MuJoCo convention: [w, x, y, z].
        (scipy uses [x, y, z, w] — do not mix them up.)
        """
        t = np.asarray(target_quat, dtype=float)
        c = np.asarray(current_quat, dtype=float)

        # conjugate of current
        c_conj = np.array([c[0], -c[1], -c[2], -c[3]])
        # q_rel = t * conj(c)
        q_rel = self._quat_mul(t, c_conj)

        # shortest-path: flip sign if scalar part is negative
        if q_rel[0] < 0:
            q_rel = -q_rel

        # small-angle approximation
        return 2.0 * q_rel[1:]

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])


# --------------------------------------------------------------------------
if __name__ == "__main__":
    solver = DiffIKSolver(damping=0.05)
    rng = np.random.default_rng(0)
    fake_J = rng.normal(size=(6, 7))

    q_now = np.zeros(7)
    pos_now = np.array([0.5, 0.0, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    pos_target = pos_now + np.array([0.01, 0.0, 0.0])

    q_target = solver.solve(pos_now, quat, pos_target, quat, fake_J, q_now)
    dq = q_target - q_now
    print("dq        :", np.round(dq, 5))
    print("norm(dq)  :", round(float(np.linalg.norm(dq)), 6))

    # verify the step actually reduces EE error (predicted via the Jacobian)
    predicted = fake_J @ dq
    err = np.concatenate([pos_target - pos_now, np.zeros(3)])
    print("err before:", round(float(np.linalg.norm(err)), 6))
    print("err after :", round(float(np.linalg.norm(err - predicted)), 6))

    # identical quaternions must give zero rotational error
    print("zero-rot check:", np.allclose(solver._quat_error(quat, quat), 0.0))