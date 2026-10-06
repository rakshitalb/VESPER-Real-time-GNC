"""
VESPER Programme - Week 10 Robustness Study

Module: week10_robustness.py

Purpose:
    Week 10 robustness, sensitivity, convergence, Monte Carlo,
    and edge-case analysis using the existing RK45 simulation.

Important:
    - Original simulator.py is not modified.
    - Original control.py is not modified.
    - Existing Week 8 convergence study is preserved.
    - Week 10 tests are added below the existing convergence study.

Reference:
    VESPER-W3-SAT-SIM-001
"""

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import (
    default_satellite,
    default_rw_model,
    build_state,
    state_derivative,
    quat_norm,
    quat_to_dcm,
    dcm_to_euler321
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
            dcm_to_euler321(
                quat_to_dcm(
                    quat_norm(self.q[i])
                )
            )
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

        print(
            f"  Max Euler error: {max_err:.3f}°"
        )

        h_sat = (
            np.max(np.abs(self.h))
            / self.rw_model['h_max']
            * 100
        )

        print(
            f"  Max wheel saturation: {h_sat:.1f}%"
        )

        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Core simulation function
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
        q_ref = np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ])

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

        # Maintain normalized quaternion.
        s[0:4] = quat_norm(s[0:4])

        # Keep reaction-wheel momentum within physical limits.
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
        dense_output=False
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
        scenario_name=scenario_name
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

    Week 10:
        Kp and Kd can be overridden through kwargs.
    """

    angle = np.radians(30.0)

    q0 = np.array([
        np.sin(angle / 2),
        0.0,
        0.0,
        np.cos(angle / 2)
    ])

    dt = kwargs.pop('dt', 0.5)

    # Allow Week 10 sensitivity studies to override gains.
    Kp = kwargs.pop('Kp', 0.008)
    Kd = kwargs.pop('Kd', 0.05)

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
        Kp=Kp,
        Kd=Kd,
        disturbances=disturbances,
        **kwargs
    )


def scenario_slew(
    controller_mode='pd',
    disturbances=True,
    **kwargs
):
    """
    Yaw slew:
    rotate 45° about body Z axis.

    Kp and Kd can be overridden for Week 10 studies.
    """

    angle = np.radians(45.0)

    q_ref = np.array([
        0.0,
        0.0,
        np.sin(angle / 2),
        np.cos(angle / 2)
    ])

    dt = kwargs.pop('dt', 0.5)

    Kp = kwargs.pop('Kp', 0.008)
    Kd = kwargs.pop('Kd', 0.05)

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
        Kp=Kp,
        Kd=Kd,
        disturbances=disturbances,
        **kwargs
    )


def scenario_disturbance_only(**kwargs):
    """
    No-control free-drift case.
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
# Helper functions for Week 10 analysis
# ---------------------------------------------------------------------------

def final_attitude_error(result):
    """
    Maximum absolute final Euler-angle error in degrees.
    """

    return float(
        np.max(
            np.abs(result.euler_deg[-1])
        )
    )


def maximum_wheel_saturation(result):
    """
    Maximum wheel momentum usage as percentage of h_max.
    """

    return float(
        np.max(np.abs(result.h))
        / result.rw_model['h_max']
        * 100.0
    )


def robustness_record(
    result,
    Kp=None,
    Kd=None
):
    """
    Convert a simulation result into a compact record.
    """

    final_error = final_attitude_error(result)

    max_saturation = maximum_wheel_saturation(result)

    final_rate = float(
        result.w_norm_deg[-1]
    )

    return {
        'Kp': Kp,
        'Kd': Kd,
        'final_error_deg': final_error,
        'final_rate_deg_s': final_rate,
        'max_saturation_percent': max_saturation
    }


# ---------------------------------------------------------------------------
# Normal reference scenarios
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    print(
        "VESPER W3 — Satellite Attitude Dynamics Simulation"
    )

    print(
        "Ref: VESPER-W3-SAT-SIM-001\n"
    )

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


# ===========================================================================
# WEEK 10 — ROBUSTNESS / SENSITIVITY STUDY
# ===========================================================================

print("\n")
print("=" * 70)
print("WEEK 10 ROBUSTNESS STUDY")
print("=" * 70)


# ---------------------------------------------------------------------------
# Week 10.1 — Kp sensitivity
# ---------------------------------------------------------------------------

print("\n")
print("-" * 70)
print("WEEK 10.1 — Kp SENSITIVITY")
print("-" * 70)

kp_values = [
    0.004,
    0.006,
    0.008,
    0.010,
    0.012
]

kp_results = []

for kp in kp_values:

    result = scenario_rest_to_rest(
        controller_mode='pd',
        disturbances=True,
        dt=0.5,
        Kp=kp,
        Kd=0.05,
        verbose=False
    )

    record = robustness_record(
        result,
        Kp=kp,
        Kd=0.05
    )

    kp_results.append(record)

    print(
        f"Kp={kp:.4f} | "
        f"Kd=0.0500 | "
        f"final error={record['final_error_deg']:.6f}° | "
        f"final rate={record['final_rate_deg_s']:.6f} deg/s | "
        f"max saturation={record['max_saturation_percent']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Week 10.2 — Kd sensitivity
# ---------------------------------------------------------------------------

print("\n")
print("-" * 70)
print("WEEK 10.2 — Kd SENSITIVITY")
print("-" * 70)

kd_values = [
    0.025,
    0.0375,
    0.050,
    0.0625,
    0.075
]

kd_results = []

for kd in kd_values:

    result = scenario_rest_to_rest(
        controller_mode='pd',
        disturbances=True,
        dt=0.5,
        Kp=0.008,
        Kd=kd,
        verbose=False
    )

    record = robustness_record(
        result,
        Kp=0.008,
        Kd=kd
    )

    kd_results.append(record)

    print(
        f"Kp=0.0080 | "
        f"Kd={kd:.4f} | "
        f"final error={record['final_error_deg']:.6f}° | "
        f"final rate={record['final_rate_deg_s']:.6f} deg/s | "
        f"max saturation={record['max_saturation_percent']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Week 10.3 — Two-parameter Kp/Kd sweep
# ---------------------------------------------------------------------------

print("\n")
print("-" * 70)
print("WEEK 10.3 — Kp/Kd PARAMETER SWEEP")
print("-" * 70)

kp_sweep = [
    0.006,
    0.008,
    0.010
]

kd_sweep = [
    0.0375,
    0.0500,
    0.0625
]

sweep_results = []

for kp in kp_sweep:

    for kd in kd_sweep:

        result = scenario_rest_to_rest(
            controller_mode='pd',
            disturbances=True,
            dt=0.5,
            Kp=kp,
            Kd=kd,
            verbose=False
        )

        record = robustness_record(
            result,
            Kp=kp,
            Kd=kd
        )

        sweep_results.append(record)

        print(
            f"Kp={kp:.4f} | "
            f"Kd={kd:.4f} | "
            f"final error={record['final_error_deg']:.6f}° | "
            f"final rate={record['final_rate_deg_s']:.6f} deg/s | "
            f"max saturation={record['max_saturation_percent']:.2f}%"
        )


# ---------------------------------------------------------------------------
# Week 10.4 — Monte Carlo robustness study
# ---------------------------------------------------------------------------

print("\n")
print("-" * 70)
print("WEEK 10.4 — MONTE CARLO ROBUSTNESS STUDY")
print("-" * 70)

print(
    "Fixed random seed = 42 for reproducibility."
)

rng = np.random.default_rng(42)

monte_carlo_results = []

number_of_trials = 20

for trial in range(1, number_of_trials + 1):

    # ±25% variation around nominal Kp and Kd.
    kp = 0.008 * rng.uniform(
        0.75,
        1.25
    )

    kd = 0.05 * rng.uniform(
        0.75,
        1.25
    )

    # Initial roll error varied between 20° and 40°.
    angle_deg = rng.uniform(
        20.0,
        40.0
    )

    angle = np.radians(angle_deg)

    q0 = np.array([
        np.sin(angle / 2),
        0.0,
        0.0,
        np.cos(angle / 2)
    ])

    result = run_simulation(
        scenario_name=f'Monte Carlo Trial {trial}',
        t_span=(0, 200),
        dt=0.5,
        q0=q0,
        w0=np.zeros(3),
        q_ref=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        controller_mode='pd',
        Kp=kp,
        Kd=kd,
        disturbances=True,
        verbose=False
    )

    record = robustness_record(
        result,
        Kp=kp,
        Kd=kd
    )

    record['trial'] = trial
    record['initial_angle_deg'] = angle_deg

    monte_carlo_results.append(record)

    print(
        f"Trial {trial:02d} | "
        f"initial={angle_deg:6.2f}° | "
        f"Kp={kp:.5f} | "
        f"Kd={kd:.5f} | "
        f"final error={record['final_error_deg']:.6f}° | "
        f"final rate={record['final_rate_deg_s']:.6f} deg/s | "
        f"saturation={record['max_saturation_percent']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Monte Carlo statistics
# ---------------------------------------------------------------------------

mc_errors = np.array([
    r['final_error_deg']
    for r in monte_carlo_results
])

mc_rates = np.array([
    r['final_rate_deg_s']
    for r in monte_carlo_results
])

mc_saturation = np.array([
    r['max_saturation_percent']
    for r in monte_carlo_results
])

print("\n")
print("MONTE CARLO STATISTICS")
print("-" * 70)

print(
    f"Trials                 : {number_of_trials}"
)

print(
    f"Final error mean       : {np.mean(mc_errors):.6f}°"
)

print(
    f"Final error maximum    : {np.max(mc_errors):.6f}°"
)

print(
    f"Final rate mean        : {np.mean(mc_rates):.6f} deg/s"
)

print(
    f"Final rate maximum     : {np.max(mc_rates):.6f} deg/s"
)

print(
    f"Maximum wheel saturation: {np.max(mc_saturation):.2f}%"
)


# ---------------------------------------------------------------------------
# Week 10.5 — Edge-case testing
# ---------------------------------------------------------------------------

print("\n")
print("-" * 70)
print("WEEK 10.5 — EDGE-CASE TESTS")
print("-" * 70)


edge_cases = [
    {
        'name': 'Zero attitude error',
        'angle_deg': 0.0
    },
    {
        'name': 'Small attitude error',
        'angle_deg': 0.1
    },
    {
        'name': 'Large attitude error',
        'angle_deg': 90.0
    }
]

edge_results = []

for case in edge_cases:

    angle = np.radians(
        case['angle_deg']
    )

    q0 = np.array([
        np.sin(angle / 2),
        0.0,
        0.0,
        np.cos(angle / 2)
    ])

    result = run_simulation(
        scenario_name=case['name'],
        t_span=(0, 200),
        dt=0.5,
        q0=q0,
        w0=np.zeros(3),
        q_ref=np.array([
            0.0,
            0.0,
            0.0,
            1.0
        ]),
        controller_mode='pd',
        Kp=0.008,
        Kd=0.05,
        disturbances=True,
        verbose=False
    )

    record = robustness_record(
        result,
        Kp=0.008,
        Kd=0.05
    )

    record['name'] = case['name']
    record['initial_angle_deg'] = case['angle_deg']

    edge_results.append(record)

    print(
        f"{case['name']:25s} | "
        f"initial={case['angle_deg']:6.2f}° | "
        f"final error={record['final_error_deg']:.6f}° | "
        f"final rate={record['final_rate_deg_s']:.6f} deg/s | "
        f"saturation={record['max_saturation_percent']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Week 10.6 — Robustness summary
# ---------------------------------------------------------------------------

print("\n")
print("=" * 70)
print("WEEK 10 ROBUSTNESS SUMMARY")
print("=" * 70)

print(
    "Completed analyses:"
)

print(
    "1. Existing Week 8 output-timestep convergence study"
)

print(
    "2. Kp sensitivity study"
)

print(
    "3. Kd sensitivity study"
)

print(
    "4. Kp/Kd two-parameter sweep"
)

print(
    "5. 20-trial reproducible Monte Carlo study"
)

print(
    "6. Edge-case testing"
)

print("\nWeek 10 robustness analysis completed.")