"""
VESPER Programme - Week 3 Prototype
Module: convergence_test.py
Description: Week 8 convergence study using the existing RK45 simulation.
             The original simulator.py is not modified.

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
        self.t = t
        self.states = states
        self.sat = sat
        self.rw_model = rw_model
        self.scenario_name = scenario_name

        self.q = states[:, 0:4]
        self.w = states[:, 4:7]
        self.h = states[:, 7:10]

        self.euler_deg = np.array([
            dcm_to_euler321(quat_to_dcm(quat_norm(self.q[i])))
            for i in range(len(t))
        ])

        self.h_rw_body = self.h

        self.w_norm_deg = np.degrees(
            np.linalg.norm(self.w, axis=1)
        )

        self.L_total = np.array([
            sat['I'] @ self.w[i] + self.h[i]
            for i in range(len(t))
        ])

    def summary(self):
        """Print a concise result summary."""
        print(f"\n{'=' * 60}")
        print(f"  Scenario : {self.scenario_name}")
        print(f"  Duration : {self.t[-1]:.1f} s  |  Steps : {len(self.t)}")
        print(f"  Final |ω|: {self.w_norm_deg[-1]:.4f} deg/s")

        euler_f = self.euler_deg[-1]

        print(
            f"  Final att: roll={euler_f[0]:.3f}°  "
            f"pitch={euler_f[1]:.3f}°  "
            f"yaw={euler_f[2]:.3f}°"
        )

        max_err = np.max(np.abs(self.euler_deg))
        print(f"  Max Euler error: {max_err:.3f}°")

        h_sat = (
            np.max(np.abs(self.h))
            / self.rw_model['h_max']
            * 100
        )

        print(f"  Max wheel saturation: {h_sat:.1f}%")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_simulation(
    scenario_name,
    t_span,
    dt,
    q0=None,
    w0=None,
    h0=None,
    q_ref=None,
    controller_mode='pd',
    Kp=0.05,
    Kd=0.10,
    Q=None,
    R_mat=None,
    disturbances=True,
    sat=None,
    rw_model=None,
    verbose=True
):
    """
    Run one simulation scenario.
    """

    if sat is None:
        sat = default_satellite()

    if rw_model is None:
        rw_model = default_rw_model()

    if q_ref is None:
        q_ref = np.array([0.0, 0.0, 0.0, 1.0])

    state0 = build_state(q0, w0, h0)

    controller = make_controller(
        mode=controller_mode,
        sat=sat,
        rw_model=rw_model,
        q_ref=q_ref,
        Kp=Kp,
        Kd=Kd,
        Q=Q,
        R_mat=R_mat
    )

    t_eval = np.arange(
        t_span[0],
        t_span[1] + dt,
        dt
    )

    if verbose:
        print(
            f"[SIM] Running '{scenario_name}' | "
            f"controller={controller_mode} | "
            f"disturbances={disturbances} | "
            f"duration={t_span[1] - t_span[0]:.0f}s | "
            f"dt={dt}s"
        )

    def rhs(t, state):

        s = state.copy()

        s[0:4] = quat_norm(s[0:4])

        h_max = rw_model['h_max']

        s[7:10] = np.clip(
            s[7:10],
            -h_max,
            h_max
        )

        return state_derivative(
            t,
            s,
            sat,
            rw_model,
            controller,
            disturbances=disturbances
        )

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
        print(
            f"[SIM] WARNING: solver did not converge — "
            f"{sol.message}"
        )

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

def scenario_rest_to_rest(
    controller_mode='pd',
    disturbances=True,
    **kwargs
):
    """
    Rest-to-rest manoeuvre:
    satellite starts tilted 30° about body X,
    controller drives it to nadir-pointing.
    """

    angle = np.radians(30.0)

    q0 = np.array([
        np.sin(angle / 2),
        0.0,
        0.0,
        np.cos(angle / 2)
    ])

    # Allow convergence study to select dt.
    dt = kwargs.pop('dt', 0.5)

    return run_simulation(
        scenario_name=f'Rest-to-Rest 30° ({controller_mode.upper()})',
        t_span=(0, 200),
        dt=dt,
        q0=q0,
        w0=np.zeros(3),
        q_ref=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        controller_mode=controller_mode,
        Kp=0.008,
        Kd=0.05,
        disturbances=disturbances,
        **kwargs
    )


def scenario_slew(
    controller_mode='pd',
    disturbances=True,
    **kwargs
):
    """
    Slew manoeuvre:
    rotate 45° about body Z axis.
    """

    angle = np.radians(45.0)

    q_ref = np.array([
        0.0,
        0.0,
        np.sin(angle / 2),
        np.cos(angle / 2)
    ])

    dt = kwargs.pop('dt', 0.5)

    return run_simulation(
        scenario_name=f'Yaw Slew 45° ({controller_mode.upper()})',
        t_span=(0, 300),
        dt=dt,
        q0=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        w0=np.zeros(3),
        q_ref=q_ref,
        controller_mode=controller_mode,
        Kp=0.008,
        Kd=0.05,
        disturbances=disturbances,
        **kwargs
    )


def scenario_disturbance_only(**kwargs):
    """
    No control:
    free drift under gravity gradient and solar pressure.
    """

    dt = kwargs.pop('dt', 1.0)

    return run_simulation(
        scenario_name='Free Drift (No Control)',
        t_span=(0, 600),
        dt=dt,
        q0=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        w0=np.array([
            0.001,
            0.0,
            0.0
        ]),
        q_ref=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        controller_mode='zero',
        disturbances=True,
        **kwargs
    )


def scenario_pd_vs_lqr(**kwargs):
    """
    Runs the same rest-to-rest manoeuvre
    with PD and LQR.
    """

    pd_res = scenario_rest_to_rest(
        controller_mode='pd',
        **kwargs
    )

    lqr_res = scenario_rest_to_rest(
        controller_mode='lqr',
        **kwargs
    )

    return pd_res, lqr_res


# ---------------------------------------------------------------------------
# Normal reference scenarios
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    print("VESPER W3 — Satellite Attitude Dynamics Simulation")
    print("Ref: VESPER-W3-SAT-SIM-001\n")

    scenario_disturbance_only()

    scenario_rest_to_rest(
        controller_mode='pd'
    )

    scenario_rest_to_rest(
        controller_mode='lqr'
    )

    scenario_slew(
        controller_mode='pd'
    )

    scenario_slew(
        controller_mode='lqr'
    )


# ---------------------------------------------------------------------------
# WEEK 8 — CONVERGENCE STUDY
# ---------------------------------------------------------------------------

print("\n")
print("=" * 70)
print("WEEK 8 CONVERGENCE STUDY — 30° REST-TO-REST PD")
print("=" * 70)

dt_values = [
    1.0,
    0.5,
    0.25,
    0.125
]

convergence_results = []

for dt in dt_values:

    result = scenario_rest_to_rest(
        controller_mode='pd',
        disturbances=True,
        dt=dt,
        verbose=False
    )

    final_attitude = result.euler_deg[-1]

    final_roll = final_attitude[0]
    final_pitch = final_attitude[1]
    final_yaw = final_attitude[2]

    convergence_results.append({
        'dt': dt,
        'roll': final_roll,
        'pitch': final_pitch,
        'yaw': final_yaw
    })

    print(
        f"dt = {dt:>5.3f} s | "
        f"roll = {final_roll:>12.8f}° | "
        f"pitch = {final_pitch:>12.8f}° | "
        f"yaw = {final_yaw:>12.8f}°"
    )


# ---------------------------------------------------------------------------
# Compare consecutive timestep results
# ---------------------------------------------------------------------------

print("\n")
print("CONSECUTIVE-STEP DIFFERENCES")
print("-" * 70)

for i in range(1, len(convergence_results)):

    previous = convergence_results[i - 1]
    current = convergence_results[i]

    roll_diff = abs(
        current['roll'] - previous['roll']
    )

    pitch_diff = abs(
        current['pitch'] - previous['pitch']
    )

    yaw_diff = abs(
        current['yaw'] - previous['yaw']
    )

    max_difference = max(
        roll_diff,
        pitch_diff,
        yaw_diff
    )

    print(
        f"{previous['dt']:.3f} s → "
        f"{current['dt']:.3f} s | "
        f"max final-attitude difference = "
        f"{max_difference:.10f}°"
    )


print("\nConvergence study completed.")