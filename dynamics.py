"""
VESPER Programme - Week 3 Prototype
Module: dynamics.py
Description: Satellite rigid-body attitude dynamics
             - Quaternion kinematics (no gimbal lock)
             - Euler rotational equations with reaction wheel coupling
             - Environmental disturbance torques
             - 3 orthogonal reaction wheels (X, Y, Z axes)

Author   : Adarsh (VESPER Phase I)
Programme: VISVAMBHARA (VESPER), EduRankAI
Ref      : VESPER-W3-SAT-SIM-001
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_norm(q):
    """Normalise a quaternion q = [qx, qy, qz, qw]."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q


def quat_multiply(p, q):
    """Hamilton product p ⊗ q. Convention: q = [qx, qy, qz, qw] (scalar last)."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array([
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
        pw*qw - px*qx - py*qy - pz*qz,
    ])


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_dcm(q):
    """Quaternion [qx,qy,qz,qw] → DCM (body from inertial)."""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),    2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),    1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


def dcm_to_euler321(R):
    """DCM → (roll, pitch, yaw) in degrees (ZYX sequence)."""
    pitch = -np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    roll  =  np.arctan2(R[1, 2], R[2, 2])
    yaw   =  np.arctan2(R[0, 1], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def omega_matrix(w):
    """4×4 matrix Ω(ω) for quaternion kinematics: dq/dt = 0.5 Ω(ω) q."""
    wx, wy, wz = w
    return np.array([
        [ 0,   wz, -wy,  wx],
        [-wz,  0,   wx,  wy],
        [ wy, -wx,  0,   wz],
        [-wx, -wy, -wz,  0 ],
    ])


# ---------------------------------------------------------------------------
# Satellite parameters
# ---------------------------------------------------------------------------

def default_satellite():
    """6U CubeSat parameters."""
    I = np.diag([0.05, 0.04, 0.06])
    return {
        'I':     I,
        'I_inv': np.linalg.inv(I),
        'mass':  6.0,
        'name':  'VESPER-SAT-6U',
    }


# ---------------------------------------------------------------------------
# Reaction wheel model: 3 orthogonal wheels aligned with body X, Y, Z
# ---------------------------------------------------------------------------

def default_rw_model():
    """
    3 orthogonal reaction wheels.
    h_rw = [h_x, h_y, h_z] angular momentum along body axes.
    Torque applied to satellite = -h_dot (equal and opposite).
    """
    return {
        'A':        np.eye(3),      # 3×3 identity: wheels aligned with body axes
        'h_max':    0.100,          # [N·m·s] per wheel
        'tau_max':  0.010,          # [N·m]   per wheel
        'n_wheels': 3,
    }


# ---------------------------------------------------------------------------
# Disturbance torques
# ---------------------------------------------------------------------------

def gravity_gradient_torque(q, I, mu=3.986e14, r=6.771e6):
    """Gravity gradient torque in body frame."""
    n = np.sqrt(mu / r**3)
    R = quat_to_dcm(q)
    nadir_body = R @ np.array([0.0, 0.0, -1.0])
    return 3.0 * n**2 * np.cross(nadir_body, I @ nadir_body)


def solar_pressure_torque(q, area=0.03, Cr=1.5,
                          P0=4.56e-6,
                          cog_offset=None):
    """Approximate solar radiation pressure torque."""
    if cog_offset is None:
        cog_offset = np.array([0.01, 0.0, 0.0])
    R = quat_to_dcm(q)
    sun_body = R @ np.array([1.0, 0.0, 0.0])
    F_srp = -P0 * Cr * area * sun_body
    return np.cross(cog_offset, F_srp)


# ---------------------------------------------------------------------------
# State derivative
# ---------------------------------------------------------------------------

def state_derivative(t, state, sat, rw_model, controller, disturbances=True):
    """
    State vector (11 elements):
        state[0:4]  = quaternion [qx,qy,qz,qw]
        state[4:7]  = body angular velocity ω [rad/s]
        state[7:10] = reaction wheel angular momenta h_rw [N·m·s]

    Returns dstate/dt (11 elements).
    """
    q = quat_norm(state[0:4])
    w = state[4:7]
    h = state[7:10]                     # wheel momenta [N·m·s], body frame

    I     = sat['I']
    I_inv = sat['I_inv']

    # wheel angular momentum in body frame (A=I so h_rw_body = h directly)
    h_rw_body = h                       # [N·m·s]

    # control torque commanded to wheels (3-vector, body frame)
    tau_cmd = controller(t, q, w, h, sat, rw_model)   # [N·m]

    # clamp to torque limit
    tau_max = rw_model['tau_max']
    tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)

    # torque ON satellite body (controller outputs desired satellite torque)
    # wheel spins in opposite direction: h_dot = -tau_cmd (absorbed by wheel)
    tau_rw = tau_cmd                    # [N·m] (controller convention: torque on body)

    # external disturbances
    tau_ext = np.zeros(3)
    if disturbances:
        tau_ext += gravity_gradient_torque(q, I)
        tau_ext += solar_pressure_torque(q)

    # Euler rotational equation: I·ω̇ = τ_rw + τ_ext − ω×(I·ω + h_rw)
    gyro  = np.cross(w, I @ w + h_rw_body)
    w_dot = I_inv @ (tau_rw + tau_ext - gyro)

    # quaternion kinematics: q̇ = 0.5 Ω(ω) q
    q_dot = 0.5 * omega_matrix(w) @ q

    # wheel absorbs opposite angular momentum: ḣ = -τ_cmd
    h_dot = -tau_cmd

    return np.concatenate([q_dot, w_dot, h_dot])


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def build_state(q0=None, w0=None, h0=None):
    """Assemble 10-element initial state vector."""
    if q0 is None: q0 = np.array([0., 0., 0., 1.])
    if w0 is None: w0 = np.zeros(3)
    if h0 is None: h0 = np.zeros(3)
    return np.concatenate([quat_norm(q0), w0, h0])
