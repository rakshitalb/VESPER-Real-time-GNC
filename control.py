"""
VESPER Programme - Week 3 Prototype
Module: control.py
Description: Attitude control laws — PD and LQR.
             3-wheel orthogonal reaction wheel configuration.

Author   : Adarsh (VESPER Phase I)
Programme: VISVAMBHARA (VESPER), EduRankAI
Ref      : VESPER-W3-SAT-SIM-001
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from dynamics import quat_multiply, quat_conjugate, quat_norm


def quaternion_error(q, q_ref):
    """Error quaternion q_err = q_ref_conj ⊗ q. Short-arc guaranteed."""
    q_err = quat_multiply(quat_conjugate(q_ref), q)
    if q_err[3] < 0:
        q_err = -q_err
    return q_err


class PDController:
    """
    PD quaternion attitude controller.
    tau_cmd [N·m] = -Kp * q_err_vec - Kd * omega
    For 3 orthogonal wheels: tau_cmd directly equals per-wheel torque vector.
    """
    def __init__(self, Kp=0.008, Kd=0.05, q_ref=None):
        self.Kp = Kp * np.ones(3)
        self.Kd = Kd * np.ones(3)
        self.q_ref = quat_norm(np.array(q_ref) if q_ref is not None
                               else np.array([0.,0.,0.,1.]))

    def set_reference(self, q_ref):
        self.q_ref = quat_norm(np.array(q_ref))

    def __call__(self, t, q, w, h, sat, rw_model):
        q_err = quaternion_error(q, self.q_ref)
        tau   = -self.Kp * q_err[:3] - self.Kd * w
        return np.clip(tau, -rw_model['tau_max'], rw_model['tau_max'])


class LQRController:
    """
    LQR attitude controller. State = [delta_q_vec(3), omega(3)].
    """
    def __init__(self, Q=None, R_mat=None, sat=None, q_ref=None):
        self.Q = Q if Q is not None else np.diag([5.,5.,5.,0.5,0.5,0.5])
        self.R = R_mat if R_mat is not None else np.diag([50.,50.,50.])
        self.q_ref = quat_norm(np.array(q_ref) if q_ref is not None
                               else np.array([0.,0.,0.,1.]))
        self.K    = None
        self._sat = sat

    def set_reference(self, q_ref):
        self.q_ref = quat_norm(np.array(q_ref))
        self.K = None

    def precompute(self, sat):
        I_inv = sat['I_inv']
        A = np.zeros((6,6))
        A[0:3,3:6] = 0.5 * np.eye(3)
        B = np.zeros((6,3))
        B[3:6,:] = I_inv
        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ B.T @ P
        except Exception:
            self.K = np.hstack([np.diag([0.008,0.008,0.008]),
                                 np.diag([0.05,0.05,0.05])])

    def __call__(self, t, q, w, h, sat, rw_model):
        if self.K is None:
            self.precompute(sat)
        q_err = quaternion_error(q, self.q_ref)
        x     = np.concatenate([q_err[:3], w])
        tau   = -self.K @ x
        return np.clip(tau, -rw_model['tau_max'], rw_model['tau_max'])


class ZeroController:
    def __call__(self, t, q, w, h, sat, rw_model):
        return np.zeros(3)


def make_controller(mode, sat, rw_model, q_ref=None,
                    Kp=0.008, Kd=0.05, Q=None, R_mat=None):
    if q_ref is None:
        q_ref = np.array([0.,0.,0.,1.])
    if mode == 'pd':
        return PDController(Kp=Kp, Kd=Kd, q_ref=q_ref)
    elif mode == 'lqr':
        ctrl = LQRController(Q=Q, R_mat=R_mat, sat=sat, q_ref=q_ref)
        ctrl.precompute(sat)
        return ctrl
    elif mode == 'zero':
        return ZeroController()
    else:
        raise ValueError(f"Unknown controller mode '{mode}'. "
                         f"Choose: 'pd', 'lqr', 'zero'.")
