"""
VESPER Programme - Week 6
Module: magnetorquer.py
Description: Magnetic actuation subsystem.
               1. Tilted-dipole Earth magnetic field model (IGRF degree-1)
               2. Three-axis magnetorquer actuator with dipole saturation
               3. B-dot detumbling control law
               4. Cross-product reaction wheel momentum dumping (desaturation)

Physics
-------
Magnetic torque:            tau = m x B          [N.m]
Tilted dipole field:        B(r) = (B0 Re^3 / r^3) [3(m_hat . r_hat) r_hat - m_hat]

Fundamental constraint
----------------------
tau = m x B is ALWAYS perpendicular to B. No magnetic torque can ever be
produced about the local field direction, so instantaneous three-axis control
is impossible. Controllability is recovered only because B rotates in the body
frame over an orbit, making the system controllable *on average*. This is why
desaturation is slow (fractions of an orbit) and must run continuously.

Momentum dumping law
--------------------
    m = (k / |B|^2) * (h x B)       ->      tau = -k * h_perp

where h_perp is the component of wheel momentum perpendicular to B.
Derivation:
    m x B = (k/|B|^2) (h x B) x B
          = (k/|B|^2) [ -h(B.B) + B(B.h) ]        (BAC-CAB)
          = -k [ h - B(B.h)/|B|^2 ]
          = -k h_perp                                          QED

Conventions (matching dynamics.py / orbit.py)
---------------------------------------------
  quaternion   q = [qx, qy, qz, qw]  (scalar last), body-from-inertial
  quat_to_dcm  returns R such that  v_body = R @ v_inertial
  frames       ECI (inertial), ECEF (Earth-fixed), body

Author   : Adarsh (VESPER Phase I)
Programme: VISVAMBHARA (VESPER), EduRankAI
Ref      : VESPER-W6-MTQ-001
"""

import numpy as np

from dynamics import quat_to_dcm, quat_norm


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def eci_to_body(v_eci, q):
    """
    Rotate an ECI vector into the body frame.

    IMPORTANT - convention note. dynamics.omega_matrix implements
    q_dot = 0.5 * q (x) omega_body, which is the kinematics for a quaternion
    representing INERTIAL-FROM-BODY. Consequently quat_to_dcm(q) returns R with

        v_inertial = R @ v_body          (NOT v_body = R @ v_inertial)

    despite the docstring in dynamics.py claiming the opposite. The inverse
    rotation is therefore the transpose. This was invisible in W3-W5 because
    every scenario regulated to the identity attitude, where R = R.T = I.
    It matters here because nadir tracking holds the body at a large,
    continuously changing rotation relative to ECI.
    """
    R = quat_to_dcm(quat_norm(q))       # inertial <- body
    return R.T @ np.asarray(v_eci, dtype=float)


def body_to_eci(v_body, q):
    """Rotate a body-frame vector into ECI. See eci_to_body for the convention."""
    R = quat_to_dcm(quat_norm(q))
    return R @ np.asarray(v_body, dtype=float)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R_EARTH = 6.3781e6        # Earth mean radius [m]
OMEGA_E = 7.2921150e-5    # Earth rotation rate [rad/s]

# Earth dipole field strength at the magnetic equator, at r = R_EARTH.
# B0 = mu0 * m_E / (4 pi Re^3), with m_E ~ 7.75e22 A.m^2
B0_DIPOLE = 3.12e-5       # [T] = 31.2 microtesla

# Geomagnetic north dip pole location (IGRF epoch ~2020)
GEOMAG_POLE_LAT_DEG =  80.4    # [deg N]
GEOMAG_POLE_LON_DEG = -72.6    # [deg E]  (i.e. 72.6 W)


# ---------------------------------------------------------------------------
# Earth magnetic field: tilted dipole model
# ---------------------------------------------------------------------------

def dipole_axis_ecef():
    """
    Unit vector of the Earth's magnetic DIPOLE MOMENT in the ECEF frame.

    The dipole moment points toward the geomagnetic SOUTH pole, which is why
    field lines emerge in the southern hemisphere and re-enter in the north.
    It is therefore the negative of the direction of the north dip pole.

    Returns
    -------
    m_hat : array(3) -- unit dipole moment direction in ECEF
    """
    lat = np.radians(GEOMAG_POLE_LAT_DEG)
    lon = np.radians(GEOMAG_POLE_LON_DEG)

    # Direction of the NORTH dip pole in ECEF
    north_pole = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])
    # Dipole moment points the other way
    return -north_pole


def eci_to_ecef_matrix(t, theta_g0=0.0):
    """
    Rotation matrix ECI -> ECEF for a simple Earth rotating about the ECI z-axis.

    Parameters
    ----------
    t        : float -- seconds since epoch
    theta_g0 : float -- Greenwich sidereal angle at epoch [rad]

    Returns
    -------
    R : array(3,3) -- v_ecef = R @ v_eci
    """
    th = theta_g0 + OMEGA_E * t
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def magnetic_field_eci(r_eci, t, theta_g0=0.0):
    """
    Earth magnetic field at an ECI position, using a tilted dipole model.

        B = (B0 Re^3 / r^3) [ 3 (m_hat . r_hat) r_hat - m_hat ]

    The dipole axis is fixed in ECEF and therefore rotates in ECI with the
    Earth. That rotation, combined with the orbital motion, is what makes the
    field direction sweep through the body frame over an orbit and restores
    average three-axis controllability.

    Parameters
    ----------
    r_eci    : array(3) -- ECI position [m]
    t        : float    -- time since epoch [s]
    theta_g0 : float    -- Greenwich sidereal angle at epoch [rad]

    Returns
    -------
    B_eci : array(3) -- magnetic field in ECI frame [T]
    """
    r_eci = np.asarray(r_eci, dtype=float)
    r = np.linalg.norm(r_eci)
    if r < 1e-6:
        return np.zeros(3)

    r_hat = r_eci / r

    # Dipole axis: fixed in ECEF, expressed in ECI
    R_eci2ecef = eci_to_ecef_matrix(t, theta_g0)
    m_hat_ecef = dipole_axis_ecef()
    m_hat_eci  = R_eci2ecef.T @ m_hat_ecef        # transpose = ECEF -> ECI

    scale = B0_DIPOLE * (R_EARTH / r) ** 3
    return scale * (3.0 * np.dot(m_hat_eci, r_hat) * r_hat - m_hat_eci)


def magnetic_field_body(r_eci, q, t, theta_g0=0.0):
    """
    Earth magnetic field expressed in the spacecraft BODY frame.

    Parameters
    ----------
    r_eci    : array(3) -- ECI position [m]
    q        : array(4) -- attitude quaternion [qx,qy,qz,qw], body-from-inertial
    t        : float    -- time since epoch [s]
    theta_g0 : float    -- Greenwich sidereal angle at epoch [rad]

    Returns
    -------
    B_body : array(3) -- magnetic field in body frame [T]
    """
    B_eci = magnetic_field_eci(r_eci, t, theta_g0)
    return eci_to_body(B_eci, q)


# ---------------------------------------------------------------------------
# Magnetorquer actuator model
# ---------------------------------------------------------------------------

def default_mtq_model():
    """
    Three orthogonal magnetorquer coils aligned with the body X, Y, Z axes.

    Values are representative of a 6U CubeSat air-core/ferrite rod set.
    """
    return {
        'A':       np.eye(3),   # coil axes (identity: aligned with body axes)
        'm_max':   0.20,        # [A.m^2] dipole limit PER AXIS
        'n_coils': 3,
        'power_per_Am2': 0.35,  # [W per A.m^2] for power bookkeeping
    }


class Magnetorquer:
    """
    Three-axis magnetorquer actuator.

    Applies per-axis dipole saturation, then produces the physical torque
    tau = m x B. Saturation matters: without it, a controller will happily
    command dipoles that no real coil can produce, and the simulated
    desaturation performance becomes fictitious.
    """

    def __init__(self, mtq_model=None):
        if mtq_model is None:
            mtq_model = default_mtq_model()
        self.model = mtq_model
        self.m_max = mtq_model['m_max']

    def saturate(self, m_cmd):
        """Clamp commanded dipole to the per-axis coil limit [A.m^2]."""
        return np.clip(np.asarray(m_cmd, dtype=float),
                       -self.m_max, self.m_max)

    def torque(self, m_cmd, B_body):
        """
        Physical torque produced on the body [N.m].

        Parameters
        ----------
        m_cmd  : array(3) -- commanded magnetic dipole [A.m^2], body frame
        B_body : array(3) -- magnetic field [T], body frame

        Returns
        -------
        tau : array(3) -- magnetic torque in body frame [N.m]
        """
        m = self.saturate(m_cmd)
        return np.cross(m, np.asarray(B_body, dtype=float))

    def power(self, m_cmd):
        """Approximate electrical power draw [W] for a commanded dipole."""
        m = self.saturate(m_cmd)
        return self.model['power_per_Am2'] * np.sum(np.abs(m))

    def is_saturated(self, m_cmd, tol=1e-9):
        """True on any axis where the command is at or beyond the coil limit."""
        m = np.abs(np.asarray(m_cmd, dtype=float))
        return m >= (self.m_max - tol)


# ---------------------------------------------------------------------------
# B-dot detumbling controller
# ---------------------------------------------------------------------------

class BDotController:
    """
    B-dot detumbling law:   m = -k * dB/dt

    Used after separation from the launch vehicle, when body rates are large
    and no attitude knowledge is available. It needs only a magnetometer, no
    gyro and no attitude solution, which is what makes it the standard
    first-mode controller on almost every LEO smallsat.

    Why it works: in the body frame, dB/dt is dominated by -omega x B when the
    spacecraft is tumbling. Commanding m proportional to -dB/dt therefore
    produces a torque that consistently opposes the rotation and bleeds off
    kinetic energy. It cannot null rates about the instantaneous field
    direction, but the field sweeps over an orbit, so all axes damp eventually.

    The derivative is computed by finite difference on successive samples with
    a first-order low-pass filter, because raw magnetometer differencing is
    extremely noisy.
    """

    def __init__(self, k=5e4, tau_filter=2.0, m_max=0.20):
        """
        Parameters
        ----------
        k          : gain [A.m^2 / (T/s)]
        tau_filter : low-pass time constant on dB/dt [s]
        m_max      : dipole limit for internal clamping [A.m^2]
        """
        self.k          = k
        self.tau_filter = tau_filter
        self.m_max      = m_max
        self._B_prev    = None
        self._t_prev    = None
        self._Bdot_f    = np.zeros(3)

    def reset(self):
        self._B_prev = None
        self._t_prev = None
        self._Bdot_f = np.zeros(3)

    def __call__(self, t, B_body):
        """
        Compute the commanded dipole [A.m^2].

        Parameters
        ----------
        t      : float    -- current time [s]
        B_body : array(3) -- measured field in body frame [T]
        """
        B_body = np.asarray(B_body, dtype=float)

        if self._B_prev is None or self._t_prev is None:
            self._B_prev = B_body.copy()
            self._t_prev = t
            return np.zeros(3)

        dt = t - self._t_prev
        if dt <= 1e-9:
            return np.clip(-self.k * self._Bdot_f, -self.m_max, self.m_max)

        Bdot_raw = (B_body - self._B_prev) / dt

        # First-order low-pass filter
        alpha = dt / (self.tau_filter + dt)
        self._Bdot_f = (1.0 - alpha) * self._Bdot_f + alpha * Bdot_raw

        self._B_prev = B_body.copy()
        self._t_prev = t

        m = -self.k * self._Bdot_f
        return np.clip(m, -self.m_max, self.m_max)


def bdot_rate_law(omega, B_body, k=1.0e-4, m_max=0.20):
    """
    Rate-based B-dot variant:   m = k * (omega x B) / |B|^2

    SIGN. The classic law is m = -k_d * dB/dt. In the body frame of a tumbling
    spacecraft dB/dt is dominated by -omega x B, so

        m = -k_d * (-omega x B) = +k_d * (omega x B)

    the sign is POSITIVE. Verify by expanding the resulting torque:

        tau = m x B = (k/|B|^2) (omega x B) x B
                    = (k/|B|^2) [ -omega(B.B) + B(B.omega) ]
                    = -k [ omega - B(B.omega)/|B|^2 ]
                    = -k * omega_perp                                    QED

    which opposes the rotation. Flipping this sign yields tau = +k*omega_perp,
    i.e. positive feedback that spins the spacecraft UP rather than down.

    Why this form and not the finite-difference BDotController inside an ODE:
    solve_ivp evaluates the RHS at non-monotonic trial times within each RK
    stage, so any controller holding state between calls (a previous sample, a
    filter) is invalid there. This algebraic form is the correct choice for
    continuous simulation; BDotController is the discrete-time implementation
    that represents what actually runs on the flight computer.

    Gain scale: |m| ~ k*|omega|/|B|. For m_max = 0.2 A.m^2 at |omega| = 5 deg/s
    and |B| = 30 uT, k ~ 1e-4 puts the command near the coil limit, which is the
    intended behaviour -- B-dot is normally bang-bang during early detumbling.

    Parameters
    ----------
    omega  : array(3) -- body angular velocity [rad/s]
    B_body : array(3) -- field in body frame [T]
    k      : gain [A.m^2 . T / (rad/s)]
    m_max  : dipole limit [A.m^2]
    """
    B = np.asarray(B_body, dtype=float)
    B_sq = np.dot(B, B)
    if B_sq < 1e-20:
        return np.zeros(3)
    m = (k / B_sq) * np.cross(np.asarray(omega, dtype=float), B)
    return np.clip(m, -m_max, m_max)


# ---------------------------------------------------------------------------
# Reaction wheel momentum dumping (desaturation)
# ---------------------------------------------------------------------------

class MomentumDumper:
    """
    Cross-product momentum dumping law:

        m = (k / |B|^2) * (h x B)        ->    tau_mag = -k * h_perp

    The magnetorquer torque acts on the BODY, not on the wheel. The wheel is
    unloaded indirectly: tau_mag perturbs the attitude, the attitude controller
    reacts by commanding wheel torque in the opposite sense, and since
    h_dot = -tau_cmd the wheel momentum bleeds away. Total system angular
    momentum L = I.omega + h is what the magnetorquer actually changes; with
    the attitude loop holding omega near zero, that change appears almost
    entirely in h.

    A deadband is included because running the coils continuously at low
    momentum wastes power and injects needless attitude disturbance for no
    benefit.
    """

    def __init__(self, k=2.0e-4, m_max=0.20, h_deadband=0.0,
                 h_target=None):
        """
        Parameters
        ----------
        k          : dumping gain [1/s]. The momentum time constant is ~1/k,
                     so k = 2e-4 gives ~5000 s, roughly one orbit.
        m_max      : per-axis dipole limit [A.m^2]
        h_deadband : below this |h| [N.m.s], command zero dipole
        h_target   : array(3) bias momentum to hold; default zero
        """
        self.k          = k
        self.m_max      = m_max
        self.h_deadband = h_deadband
        self.h_target   = np.zeros(3) if h_target is None \
                          else np.asarray(h_target, dtype=float)

    def __call__(self, h, B_body):
        """
        Compute the commanded dipole [A.m^2].

        Parameters
        ----------
        h      : array(3) -- reaction wheel momentum, body frame [N.m.s]
        B_body : array(3) -- magnetic field, body frame [T]
        """
        h = np.asarray(h, dtype=float)
        B = np.asarray(B_body, dtype=float)

        h_err = h - self.h_target

        if np.linalg.norm(h_err) < self.h_deadband:
            return np.zeros(3)

        B_sq = np.dot(B, B)
        if B_sq < 1e-20:
            return np.zeros(3)

        m = (self.k / B_sq) * np.cross(h_err, B)
        return np.clip(m, -self.m_max, self.m_max)

    def predicted_torque(self, h, B_body):
        """
        Ideal (unsaturated) torque this law is trying to produce, -k * h_perp.
        Useful for verifying that the achieved torque matches theory when the
        coils are not saturated.
        """
        h = np.asarray(h, dtype=float) - self.h_target
        B = np.asarray(B_body, dtype=float)
        B_sq = np.dot(B, B)
        if B_sq < 1e-20:
            return np.zeros(3)
        h_perp = h - B * np.dot(B, h) / B_sq
        return -self.k * h_perp


# ---------------------------------------------------------------------------
# Combined magnetic control mode manager
# ---------------------------------------------------------------------------

class MagneticControlSystem:
    """
    Chooses between detumbling and momentum dumping, and produces the resulting
    magnetic torque on the body.

    Modes
    -----
    'detumble'  : B-dot only. Used when |omega| exceeds a threshold.
    'dump'      : momentum dumping only. Nominal on-orbit mode.
    'auto'      : detumble while tumbling, then switch to dumping. Includes
                  hysteresis so the mode does not chatter at the threshold.
    'off'       : coils disabled.
    """

    def __init__(self, mtq=None, dumper=None, bdot=None,
                 mode='auto',
                 k_bdot_rate=1.0e-4,
                 w_detumble_on=np.radians(1.0),
                 w_detumble_off=np.radians(0.2)):
        self.mtq    = mtq    if mtq    is not None else Magnetorquer()
        self.dumper = dumper if dumper is not None else MomentumDumper(
            m_max=self.mtq.m_max)
        self.bdot   = bdot   if bdot   is not None else BDotController(
            m_max=self.mtq.m_max)

        self.mode = mode
        # The rate form and the derivative form have DIFFERENT units and
        # therefore different gain scales; they are not interchangeable.
        # k_bdot_rate [A.m^2.T/(rad/s)] vs BDotController.k [A.m^2/(T/s)].
        self.k_bdot_rate = k_bdot_rate
        self.w_on  = w_detumble_on
        self.w_off = w_detumble_off
        self._detumbling = False

        # Diagnostics from the most recent call
        self.last_m      = np.zeros(3)
        self.last_tau    = np.zeros(3)
        self.last_mode   = 'off'
        self.last_B      = np.zeros(3)

    def active_mode(self, w):
        """Resolve the effective mode, applying hysteresis in 'auto'."""
        if self.mode != 'auto':
            return self.mode

        w_mag = np.linalg.norm(w)
        if self._detumbling:
            if w_mag < self.w_off:
                self._detumbling = False
        else:
            if w_mag > self.w_on:
                self._detumbling = True

        return 'detumble' if self._detumbling else 'dump'

    def __call__(self, t, r_eci, q, w, h, theta_g0=0.0):
        """
        Compute the magnetic torque applied to the body this instant.

        Parameters
        ----------
        t        : float    -- time [s]
        r_eci    : array(3) -- ECI position [m]
        q        : array(4) -- attitude quaternion [qx,qy,qz,qw]
        w        : array(3) -- body rate [rad/s]
        h        : array(3) -- wheel momentum [N.m.s]
        theta_g0 : float    -- Greenwich angle at epoch [rad]

        Returns
        -------
        tau_mag : array(3) -- magnetic torque on the body [N.m]
        """
        B_body = magnetic_field_body(r_eci, q, t, theta_g0)
        self.last_B = B_body

        mode = self.active_mode(w)
        self.last_mode = mode

        if mode == 'off':
            m = np.zeros(3)
        elif mode == 'detumble':
            m = bdot_rate_law(w, B_body, k=self.k_bdot_rate,
                              m_max=self.mtq.m_max)
        elif mode == 'dump':
            m = self.dumper(h, B_body)
        else:
            raise ValueError(f"Unknown magnetic control mode '{mode}'. "
                             f"Choose: 'auto', 'detumble', 'dump', 'off'.")

        self.last_m   = self.mtq.saturate(m)
        self.last_tau = self.mtq.torque(m, B_body)
        return self.last_tau


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def momentum_time_constant(k):
    """Approximate wheel momentum decay time constant [s] for gain k."""
    return 1.0 / k if k > 0 else np.inf


def max_available_torque(m_max, B_mag):
    """Largest magnetic torque magnitude achievable [N.m]."""
    return m_max * B_mag


def field_sweep_over_orbit(prop_r_list, t_list, q=None, theta_g0=0.0):
    """
    Field direction history, used to confirm that B genuinely rotates in the
    body frame over an orbit. If it did not, three-axis desaturation would be
    impossible rather than merely slow.

    Parameters
    ----------
    prop_r_list : list of array(3) -- ECI positions [m]
    t_list      : list of float    -- times [s]
    q           : array(4) or None -- fixed attitude; None means ECI frame
    theta_g0    : float            -- Greenwich angle at epoch [rad]

    Returns
    -------
    B_hist : array(N,3) [T]
    """
    out = []
    for r, t in zip(prop_r_list, t_list):
        if q is None:
            out.append(magnetic_field_eci(r, t, theta_g0))
        else:
            out.append(magnetic_field_body(r, q, t, theta_g0))
    return np.array(out)


if __name__ == '__main__':
    # Quick self-check
    np.set_printoptions(precision=4, suppress=False)
    print("VESPER W6 - magnetorquer.py self-check")
    print("Ref: VESPER-W6-MTQ-001\n")

    r = np.array([R_EARTH + 400e3, 0.0, 0.0])
    B = magnetic_field_eci(r, 0.0)
    print(f"  B at 400 km equator : {np.linalg.norm(B)*1e9:.1f} nT "
          f"({np.linalg.norm(B)*1e6:.2f} uT)")

    r_pole = np.array([0.0, 0.0, R_EARTH + 400e3])
    B_pole = magnetic_field_eci(r_pole, 0.0)
    print(f"  B at 400 km pole    : {np.linalg.norm(B_pole)*1e6:.2f} uT")

    mtq = Magnetorquer()
    print(f"  Max torque @ 30 uT  : "
          f"{max_available_torque(mtq.m_max, 30e-6)*1e6:.2f} uN.m")

    d = MomentumDumper(k=2e-4)
    print(f"  Dumping time const  : {momentum_time_constant(d.k):.0f} s "
          f"({momentum_time_constant(d.k)/5554:.2f} orbits)")

    h = np.array([0.01, 0.0, 0.0])
    m = d(h, B)
    tau = mtq.torque(m, B)
    print(f"  Dipole for h=[0.01,0,0]: {m} A.m^2")
    print(f"  Achieved torque        : {tau*1e6} uN.m")
    print(f"  Predicted (-k h_perp)  : {d.predicted_torque(h, B)*1e6} uN.m")
