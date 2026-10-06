"""
VESPER Programme - Week 5
Module: orbit.py
Description: Two-body Keplerian orbit propagator with J2 oblateness
             perturbation and atmospheric drag (cannonball model).
             Provides the satellite's position and velocity in the
             inertial (ECI) frame at each timestep, which feeds:
               1. Gravity gradient torque (uses true nadir direction)
               2. Atmospheric density for drag
               3. Ground-track visualisation

Physical constants and models:
  - Two-body: F = -mu/r^2 * r_hat
  - J2: dominant Earth oblateness perturbation
  - Drag: cannonball model with NRLMSISE-00-inspired density table
  - Magnetic field: tilted dipole (for magnetometer sensor model)

State vector (6 elements):
    [rx, ry, rz]   -- position  [m]  in ECI frame
    [vx, vy, vz]   -- velocity  [m/s] in ECI frame

Author   : Adarsh (VESPER Phase I)
Programme: VISVAMBHARA (VESPER), EduRankAI
Ref      : VESPER-W5-ORB-001
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU_EARTH   = 3.986004418e14   # Earth gravitational parameter [m^3/s^2]
R_EARTH    = 6.3781e6          # Earth mean radius [m]
J2         = 1.08263e-3        # J2 oblateness coefficient (dimensionless)
OMEGA_E    = 7.2921150e-5      # Earth rotation rate [rad/s]

# Atmospheric density model: simple exponential scale-height model
# Reference: Vallado, "Fundamentals of Astrodynamics and Applications", Table 8-4
# Altitude [m] : (rho0 [kg/m^3], H [m])
_ATMO_TABLE = [
    (  0e3,  1.225,      8500),
    ( 25e3,  4.008e-2,   6600),
    ( 50e3,  1.057e-3,   7700),
    ( 75e3,  3.972e-5,   8000),
    (100e3,  5.604e-7,   5900),
    (125e3,  3.508e-8,   7200),
    (150e3,  2.070e-9,   8600),
    (200e3,  2.541e-10, 10000),
    (250e3,  6.073e-11, 11000),
    (300e3,  1.916e-11, 12000),
    (350e3,  5.721e-12, 14000),
    (400e3,  2.803e-12, 17500),
    (500e3,  5.215e-13, 26000),
    (600e3,  1.137e-13, 34000),
    (700e3,  3.070e-14, 45000),
    (800e3,  1.136e-14, 53000),
    (900e3,  5.759e-15, 60000),
    (1000e3, 3.561e-15, 72000),
]


# ---------------------------------------------------------------------------
# Atmospheric density
# ---------------------------------------------------------------------------

def atmo_density(alt_m):
    """
    Exponential atmosphere model.
    Returns density [kg/m^3] at altitude alt_m [m] above Earth's surface.
    """
    alt = np.clip(alt_m, 0, 1000e3)
    for i in range(len(_ATMO_TABLE) - 1):
        h0, rho0, H = _ATMO_TABLE[i]
        h1          = _ATMO_TABLE[i + 1][0]
        if alt <= h1:
            return rho0 * np.exp(-(alt - h0) / H)
    # Above 1000 km: use last entry
    h0, rho0, H = _ATMO_TABLE[-1]
    return rho0 * np.exp(-(alt - h0) / H)


# ---------------------------------------------------------------------------
# Orbit perturbation accelerations
# ---------------------------------------------------------------------------

def accel_j2(r_vec):
    """
    J2 oblateness perturbation acceleration [m/s^2] in ECI frame.

    a_J2 = -(3/2) * J2 * mu * Re^2 / r^5 *
           [x(1 - 5z^2/r^2), y(1 - 5z^2/r^2), z(3 - 5z^2/r^2)]

    Parameters
    ----------
    r_vec : array(3) -- ECI position [m]
    """
    x, y, z = r_vec
    r       = np.linalg.norm(r_vec)
    factor  = -1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
    z2r2    = (z / r) ** 2
    ax = factor * x * (1.0 - 5.0 * z2r2)
    ay = factor * y * (1.0 - 5.0 * z2r2)
    az = factor * z * (3.0 - 5.0 * z2r2)
    return np.array([ax, ay, az])


def accel_drag(r_vec, v_vec, Cd=2.2, A_m=0.02):
    """
    Atmospheric drag acceleration (cannonball model) [m/s^2].

    a_drag = -0.5 * rho * Cd * (A/m) * |v_rel|^2 * v_rel_hat

    Parameters
    ----------
    r_vec : array(3)  -- ECI position [m]
    v_vec : array(3)  -- ECI velocity [m/s]
    Cd    : float     -- drag coefficient (2.2 for sphere)
    A_m   : float     -- area-to-mass ratio [m^2/kg]
                         (6U CubeSat: ~0.06 m^2 / 6 kg = 0.01 m^2/kg;
                          use 0.02 for conservative estimate)
    """
    alt  = np.linalg.norm(r_vec) - R_EARTH
    rho  = atmo_density(alt)

    # Velocity relative to rotating atmosphere
    omega_cross_r = np.cross(np.array([0, 0, OMEGA_E]), r_vec)
    v_rel         = v_vec - omega_cross_r
    v_mag         = np.linalg.norm(v_rel)

    if v_mag < 1e-6:
        return np.zeros(3)

    return -0.5 * rho * Cd * A_m * v_mag * v_rel


# ---------------------------------------------------------------------------
# Orbit propagator ODE
# ---------------------------------------------------------------------------

def orbit_rhs(t, state, Cd=2.2, A_m=0.02, include_j2=True,
              include_drag=True):
    """
    Right-hand side of the orbital equations of motion.

    state = [rx, ry, rz, vx, vy, vz]
    dstate/dt = [vx, vy, vz, ax, ay, az]
    """
    r_vec = state[0:3]
    v_vec = state[3:6]
    r     = np.linalg.norm(r_vec)

    # Two-body gravity
    a_grav = -MU_EARTH / r**3 * r_vec

    # J2 perturbation
    a_j2 = accel_j2(r_vec) if include_j2 else np.zeros(3)

    # Atmospheric drag
    a_drag = accel_drag(r_vec, v_vec, Cd, A_m) if include_drag \
             else np.zeros(3)

    a_total = a_grav + a_j2 + a_drag

    return np.concatenate([v_vec, a_total])


# ---------------------------------------------------------------------------
# Orbital element utilities
# ---------------------------------------------------------------------------

def circular_orbit_state(altitude_m, inclination_deg=51.6,
                          raan_deg=0.0, true_anomaly_deg=0.0):
    """
    Generate ECI state vector for a circular orbit.

    Parameters
    ----------
    altitude_m      : orbit altitude above Earth's surface [m]
    inclination_deg : inclination [deg]
    raan_deg        : right ascension of ascending node [deg]
    true_anomaly_deg: initial true anomaly [deg]

    Returns
    -------
    state0 : array(6) -- [rx,ry,rz, vx,vy,vz] in ECI [m, m/s]
    T_orb  : float    -- orbital period [s]
    n      : float    -- mean motion [rad/s]
    """
    r_orb = R_EARTH + altitude_m
    n     = np.sqrt(MU_EARTH / r_orb**3)   # mean motion [rad/s]
    v_orb = np.sqrt(MU_EARTH / r_orb)       # circular speed [m/s]
    T_orb = 2 * np.pi / n                   # period [s]

    inc  = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    nu   = np.radians(true_anomaly_deg)

    # Position in orbital plane, then rotate to ECI
    # Perifocal frame: P along periapsis, Q = 90 deg ahead
    cos_nu, sin_nu = np.cos(nu), np.sin(nu)
    r_peri = r_orb * np.array([cos_nu, sin_nu, 0.0])
    v_peri = v_orb * np.array([-sin_nu, cos_nu, 0.0])

    # Rotation matrix: perifocal → ECI (Omega, i, omega; omega=0 for circular)
    ci, si = np.cos(inc), np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)
    # Simplified rotation for omega=0 (circular orbit):
    R = np.array([
        [ cr, -sr*ci,  sr*si],
        [ sr,  cr*ci, -cr*si],
        [0.0,     si,     ci],
    ])

    r_eci = R @ r_peri
    v_eci = R @ v_peri

    return np.concatenate([r_eci, v_eci]), T_orb, n


def orbit_altitude(r_vec):
    """Return altitude above Earth's surface [m]."""
    return np.linalg.norm(r_vec) - R_EARTH


def nadir_direction_eci(r_vec):
    """Unit vector from satellite to Earth centre in ECI frame."""
    return -r_vec / np.linalg.norm(r_vec)


def nadir_direction_body(r_vec, q):
    """
    Unit vector from satellite to Earth centre in body frame.

    Parameters
    ----------
    r_vec : array(3) -- ECI position [m]
    q     : array(4) -- quaternion [qx,qy,qz,qw] (body from inertial)
    """
    from dynamics import quat_to_dcm, quat_norm
    R = quat_to_dcm(quat_norm(q))     # inertial → body
    nadir_inertial = nadir_direction_eci(r_vec)
    return R @ nadir_inertial


def orbital_frame_vectors(r_vec, v_vec):
    """
    Return the three unit vectors of the LVLH (Local Vertical Local Horizontal)
    frame in ECI coordinates.

    z_lvlh = -r_hat (nadir)
    y_lvlh = -h_hat (negative orbit normal)
    x_lvlh = y_lvlh × z_lvlh (along-track)
    """
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)

    z_lvlh = -r_hat
    y_lvlh = -h_hat
    x_lvlh = np.cross(y_lvlh, z_lvlh)
    return x_lvlh, y_lvlh, z_lvlh


# ---------------------------------------------------------------------------
# Gravity gradient torque (using real orbit geometry)
# ---------------------------------------------------------------------------

def gravity_gradient_torque_orbit(r_vec, q, I):
    """
    Gravity gradient torque in body frame using the actual satellite
    position in orbit (not a fixed orbit radius approximation).

    tau_gg = 3 * (mu/r^3) * nadir_body × (I * nadir_body)

    Parameters
    ----------
    r_vec : array(3) -- ECI position [m]
    q     : array(4) -- attitude quaternion [qx,qy,qz,qw]
    I     : array(3,3) -- inertia tensor [kg·m^2]

    Returns
    -------
    tau_gg : array(3) -- gravity gradient torque in body frame [N·m]
    """
    r     = np.linalg.norm(r_vec)
    n_sq  = MU_EARTH / r**3     # mu/r^3 [rad^2/s^2]

    nadir_b = nadir_direction_body(r_vec, q)
    tau_gg  = 3.0 * n_sq * np.cross(nadir_b, I @ nadir_b)
    return tau_gg


# ---------------------------------------------------------------------------
# Integrated propagator class
# ---------------------------------------------------------------------------

class OrbitPropagator:
    """
    Propagates a satellite orbit step by step, providing position,
    velocity, altitude, and derived quantities at each timestep.

    Usage
    -----
    prop = OrbitPropagator(altitude_m=400e3, inclination_deg=51.6)
    prop.step(dt)           # advance by dt seconds
    r, v = prop.r_eci, prop.v_eci
    """

    def __init__(self, altitude_m=400e3, inclination_deg=51.6,
                 raan_deg=0.0, true_anomaly_deg=0.0,
                 Cd=2.2, A_m=0.02,
                 include_j2=True, include_drag=True):
        """
        Initialise orbit propagator.

        Parameters
        ----------
        altitude_m      : orbit altitude [m]
        inclination_deg : inclination [deg] (ISS: 51.6, sun-sync: ~97.8)
        raan_deg        : right ascension of ascending node [deg]
        true_anomaly_deg: initial true anomaly (satellite position in orbit)
        Cd, A_m         : drag parameters
        include_j2      : include J2 oblateness perturbation
        include_drag    : include atmospheric drag
        """
        self.Cd           = Cd
        self.A_m          = A_m
        self.include_j2   = include_j2
        self.include_drag = include_drag

        state0, self.T_orb, self.n = circular_orbit_state(
            altitude_m, inclination_deg, raan_deg, true_anomaly_deg
        )
        self._state = state0.copy()

        # Logging
        self.t       = 0.0
        self.history = {'t': [], 'r': [], 'v': [], 'alt': []}
        self._log()

    def _log(self):
        self.history['t'].append(self.t)
        self.history['r'].append(self._state[0:3].copy())
        self.history['v'].append(self._state[3:6].copy())
        self.history['alt'].append(orbit_altitude(self._state[0:3]))

    def step(self, dt):
        """Advance orbit by dt seconds using RK45."""
        sol = solve_ivp(
            lambda t, s: orbit_rhs(t, s, self.Cd, self.A_m,
                                   self.include_j2, self.include_drag),
            (self.t, self.t + dt),
            self._state,
            method='RK45', rtol=1e-9, atol=1e-11,
            dense_output=False,
        )
        self._state = sol.y[:, -1]
        self.t     += dt
        self._log()

    @property
    def r_eci(self):
        """ECI position vector [m]."""
        return self._state[0:3].copy()

    @property
    def v_eci(self):
        """ECI velocity vector [m/s]."""
        return self._state[3:6].copy()

    @property
    def altitude(self):
        """Altitude above Earth surface [m]."""
        return orbit_altitude(self._state[0:3])

    @property
    def speed(self):
        """Orbital speed [m/s]."""
        return np.linalg.norm(self._state[3:6])

    @property
    def orbital_period(self):
        """Nominal orbital period [s]."""
        return self.T_orb

    @property
    def mean_motion(self):
        """Current mean motion [rad/s] based on current radius."""
        r = np.linalg.norm(self._state[0:3])
        return np.sqrt(MU_EARTH / r**3)

    def gravity_gradient_torque(self, q, I):
        """Gravity gradient torque in body frame [N·m]."""
        return gravity_gradient_torque_orbit(self._state[0:3], q, I)

    def summary(self):
        """Print a one-line orbit status."""
        r = np.linalg.norm(self._state[0:3])
        v = np.linalg.norm(self._state[3:6])
        print(f"  t={self.t:.1f}s | alt={self.altitude/1e3:.2f}km | "
              f"speed={v:.2f}m/s | n={np.degrees(self.mean_motion)*3600:.4f}°/hr")
