"""
VESPER Programme - Week 3 Prototype
Module: simulator.py
Description: Main simulation loop using scipy RK45 integrator.
             Runs configurable scenarios and returns logged data.

Author   : Adarsh (VESPER Phase I)
Programme: VISVAMBHARA (VESPER), EduRankAI
Ref      : VESPER-W3-SAT-SIM-001
"""

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import (
    default_satellite, default_rw_model,
    build_state, state_derivative,
    quat_norm, quat_to_dcm, dcm_to_euler321
)
from control import make_controller


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class SimResult:
    """Holds all logged simulation data."""

    def __init__(self, t, states, sat, rw_model, scenario_name):
        self.t            = t                       # [N] time array [s]
        self.states       = states                  # [N×13] state history
        self.sat          = sat
        self.rw_model     = rw_model
        self.scenario_name = scenario_name

        # Unpack for convenience
        self.q   = states[:, 0:4]                  # quaternion history
        self.w   = states[:, 4:7]                  # angular velocity [rad/s]
        self.h   = states[:, 7:10]                 # wheel momenta [N·m·s]  (3 wheels)

        # Derived quantities
        self.euler_deg = np.array([
            dcm_to_euler321(quat_to_dcm(quat_norm(self.q[i])))
            for i in range(len(t))
        ])                                          # [N×3] roll,pitch,yaw [deg]

        self.h_rw_body = self.h                        # [N×3] 3 wheels aligned with body axes

        self.w_norm_deg = np.degrees(
            np.linalg.norm(self.w, axis=1)
        )                                           # [N] |ω| [deg/s]

        # Angular momentum conservation check (no external torques scenario)
        self.L_total = np.array([
            sat['I'] @ self.w[i] + self.h[i]
            for i in range(len(t))
        ])                                          # [N×3] total angular momentum

    def summary(self):
        """Print a concise result summary."""
        print(f"\n{'='*60}")
        print(f"  Scenario : {self.scenario_name}")
        print(f"  Duration : {self.t[-1]:.1f} s  |  Steps : {len(self.t)}")
        print(f"  Final |ω|: {self.w_norm_deg[-1]:.4f} deg/s")
        euler_f = self.euler_deg[-1]
        print(f"  Final att: roll={euler_f[0]:.3f}°  "
              f"pitch={euler_f[1]:.3f}°  yaw={euler_f[2]:.3f}°")
        max_err = np.max(np.abs(self.euler_deg))
        print(f"  Max Euler error: {max_err:.3f}°")
        h_sat = np.max(np.abs(self.h)) / self.rw_model['h_max'] * 100  # 3 wheels
        print(f"  Max wheel saturation: {h_sat:.1f}%")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_simulation(scenario_name, t_span, dt,
                   q0=None, w0=None, h0=None,
                   q_ref=None,
                   controller_mode='pd',
                   Kp=0.05, Kd=0.10,
                   Q=None, R_mat=None,
                   disturbances=True,
                   sat=None, rw_model=None,
                   verbose=True):
    """
    Run one simulation scenario.

    Parameters
    ----------
    scenario_name    : string label for plots/logs
    t_span           : (t_start, t_end) [s]
    dt               : output time step [s]
    q0               : initial quaternion; default identity
    w0               : initial angular velocity [rad/s]; default zeros
    h0               : initial wheel momenta [N·m·s]; default zeros
    q_ref            : target quaternion; default identity (stabilise)
    controller_mode  : 'pd', 'lqr', or 'zero'
    Kp, Kd           : PD gains
    Q, R_mat         : LQR cost matrices
    disturbances     : include gravity gradient + solar pressure
    sat              : satellite dict; default 6U CubeSat
    rw_model         : reaction wheel dict; default pyramid 4-wheel
    verbose          : print progress and summary

    Returns
    -------
    SimResult object
    """
    if sat      is None: sat      = default_satellite()
    if rw_model is None: rw_model = default_rw_model()
    if q_ref    is None: q_ref    = np.array([0.0, 0.0, 0.0, 1.0])

    state0 = build_state(q0, w0, h0)

    controller = make_controller(
        mode=controller_mode, sat=sat, rw_model=rw_model,
        q_ref=q_ref, Kp=Kp, Kd=Kd, Q=Q, R_mat=R_mat
    )

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    if verbose:
        print(f"[SIM] Running '{scenario_name}' | "
              f"controller={controller_mode} | "
              f"disturbances={disturbances} | "
              f"duration={t_span[1]-t_span[0]:.0f}s")

    def rhs(t, state):
        # Work on a copy to avoid corrupting scipy's internal RK stages
        s = state.copy()
        s[0:4] = quat_norm(s[0:4])
        h_max = rw_model['h_max']
        s[7:10] = np.clip(s[7:10], -h_max, h_max)
        return state_derivative(t, s, sat, rw_model, controller,
                                disturbances=disturbances)

    sol = solve_ivp(
        rhs,
        t_span,
        state0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    if not sol.success:
        print(f"[SIM] WARNING: solver did not converge — {sol.message}")

    result = SimResult(
        t=sol.t,
        states=sol.y.T,
        sat=sat,
        rw_model=rw_model,
        scenario_name=scenario_name,
    )

    if verbose:
        result.summary()

    return result


# ---------------------------------------------------------------------------
# Pre-defined scenarios
# ---------------------------------------------------------------------------

def scenario_rest_to_rest(controller_mode='pd', disturbances=True, **kwargs):
    """
    Rest-to-rest manoeuvre: satellite starts tilted 30° about body X,
    controller drives it to nadir-pointing (identity quaternion).
    """
    angle = np.radians(30.0)
    q0 = np.array([np.sin(angle/2), 0.0, 0.0, np.cos(angle/2)])
    return run_simulation(
        scenario_name=f'Rest-to-Rest 30° ({controller_mode.upper()})',
        t_span=(0, 200),
        dt=0.5,
        q0=q0,
        w0=np.zeros(3),
        q_ref=np.array([0.0, 0.0, 0.0, 1.0]),
        controller_mode=controller_mode,
        Kp=0.008, Kd=0.05,
        disturbances=disturbances,
        **kwargs
    )


def scenario_slew(controller_mode='pd', disturbances=True, **kwargs):
    """
    Slew manoeuvre: rotate 45° about body Z axis (yaw slew).
    """
    angle = np.radians(45.0)
    q_ref = np.array([0.0, 0.0, np.sin(angle/2), np.cos(angle/2)])
    return run_simulation(
        scenario_name=f'Yaw Slew 45° ({controller_mode.upper()})',
        t_span=(0, 300),
        dt=0.5,
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        w0=np.zeros(3),
        q_ref=q_ref,
        controller_mode=controller_mode,
        Kp=0.008, Kd=0.05,
        disturbances=disturbances,
        **kwargs
    )


def scenario_disturbance_only(**kwargs):
    """
    No control: free drift under gravity gradient and solar pressure only.
    Shows why control is necessary.
    """
    return run_simulation(
        scenario_name='Free Drift (No Control)',
        t_span=(0, 600),
        dt=1.0,
        q0=np.array([0.0, 0.0, 0.0, 1.0]),
        w0=np.array([0.001, 0.0, 0.0]),   # tiny initial rate
        q_ref=np.array([0.0, 0.0, 0.0, 1.0]),
        controller_mode='zero',
        disturbances=True,
        **kwargs
    )


def scenario_pd_vs_lqr(**kwargs):
    """
    Runs the same rest-to-rest manoeuvre with PD and LQR for comparison.
    Returns (pd_result, lqr_result).
    """
    pd_res  = scenario_rest_to_rest(controller_mode='pd',  **kwargs)
    lqr_res = scenario_rest_to_rest(controller_mode='lqr', **kwargs)
    return pd_res, lqr_res


# ---------------------------------------------------------------------------
# Entry point — run all scenarios
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("VESPER W3 — Satellite Attitude Dynamics Simulation")
    print("Ref: VESPER-W3-SAT-SIM-001\n")

    scenario_disturbance_only()
    scenario_rest_to_rest(controller_mode='pd')
    scenario_rest_to_rest(controller_mode='lqr')
    scenario_slew(controller_mode='pd')
    scenario_slew(controller_mode='lqr')
