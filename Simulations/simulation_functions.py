#%%
##### IMPORTS ###############

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from tqdm import tqdm
from scipy.special import erf, erfinv
from scipy.integrate import solve_ivp as _solve_ivp_new
from joblib import Parallel as _Parallel_new, delayed as _delayed_new


##### CONSTANTS ###############
H    = const.h    # Planck's constant [J·s]
C    = const.c    # speed of light [m/s]
HBAR = const.hbar # reduced Planck's constant [J·s]
PI   = np.pi
AMU  = 1.66e-27   # atomic mass unit [kg]
kb   = const.k    # Boltzmann constant [J/K]
MU_B = const.physical_constants['Bohr magneton'][0]  # Bohr magneton [J/T]

##### TRANSITIONS ###############
gamma_689, lambda_689 = 2*PI * 7.48e3,  689.4489e-9
gamma_688, lambda_688 = 2*PI * 3.90e6,  688.020770e-9
gamma_679, lambda_679 = 2*PI * 1.26e6,  679.288943e-9
gamma_707, lambda_707 = 2*PI * 6.225e6, 707.197215e-9  # 3S1 → 3P2 (~707 nm), dominant loss channel

# Lande g-factors
G_J_3P1 = 3/2   # 3P1 excited state

# Saturation intensity: I_sat = (pi h c Gamma) / (3 lambda^3) [W/m^2] x 100 -> [uW/cm^2]
# Assumes maximum polarization coupling (sigma+ driving the cycling transition).
# For partial coupling, multiply the resulting Rabi frequency by the coupling factor.
Ipref = H * C * np.pi / 3

Is_689 = Ipref * gamma_689 / (lambda_689**3) * 100  # [uW/cm^2]
Is_688 = Ipref * gamma_688 / (lambda_688**3) * 100  # [uW/cm^2]
Is_679 = Ipref * gamma_679 / (lambda_679**3) * 100  # [uW/cm^2]


def sample_atomic_ensemble(radii, temperatures, mass=88*AMU, n_samples=1):
    """
    Sample positions and velocities from a thermal atomic ensemble.

    Positions are drawn from a Gaussian with std sigma_r. Velocities are
    drawn from a 3D Maxwell-Boltzmann distribution at temperature T.

    Args:
        radii        (float or array-like): Position std dev [m]. A scalar
                         applies the same sigma to all axes; a length-3 array
                         sets independent sigmas for x, y, z.
        temperatures (float or array-like): Temperature [K], same shape as radii.
        mass         (float): Atomic mass [kg]. Defaults to 88 amu (Sr-88).
        n_samples    (int):   Number of atoms to sample.

    Returns:
        positions  (ndarray): Shape (3,) if n_samples==1, else (n_samples, 3) [m].
        velocities (ndarray): Shape (3,) if n_samples==1, else (n_samples, 3) [m/s].
    """
    sigma_r = np.array(radii)
    sigma_v = np.sqrt(kb * np.array(temperatures) / mass)

    positions  = np.random.normal(loc=0.0, scale=sigma_r, size=(n_samples, 3))
    velocities = np.random.normal(loc=0.0, scale=sigma_v, size=(n_samples, 3))

    if n_samples == 1:
        return np.atleast_2d(positions[0]), np.atleast_2d(velocities[0])
    return np.atleast_2d(positions), np.atleast_2d(velocities)


def get_k_hat(theta, theta_z):
    """
    Return a unit wavevector for a beam at azimuthal angle theta and
    elevation angle theta_z above the x-y plane.

    Args:
        theta   (float): Azimuthal angle in the x-y plane [rad].
        theta_z (float): Elevation angle above the x-y plane [rad].

    Returns:
        ndarray: Unit vector [cos(theta_z)cos(theta), cos(theta_z)sin(theta), sin(theta_z)].
    """
    return np.array([np.cos(theta_z)*np.cos(theta),
                     np.cos(theta_z)*np.sin(theta),
                     np.sin(theta_z)])


def get_effective_r_perp(pos, k_vec):
    """
    Compute the perpendicular distance from the beam axis for each atom.

    For a beam propagating along k_vec, this returns the magnitude of each
    atom's position component in the plane normal to k_vec.

    Args:
        pos   (ndarray): Atom positions, shape (N, 3) [m].
        k_vec (ndarray): Beam wavevector (need not be normalized), shape (3,).

    Returns:
        ndarray: Perpendicular distances |r_perp|, shape (N,) [m].
    """
    k_hat      = k_vec / np.linalg.norm(k_vec)
    proj_mag   = np.sum(pos * k_hat, axis=1)   # scalar projection onto beam axis, (N,)
    r_parallel = np.outer(proj_mag, k_hat)      # parallel component, (N, 3)
    r_perp_vec = pos - r_parallel               # perpendicular component, (N, 3)
    return np.linalg.norm(r_perp_vec, axis=1)


def decompose_polarization(eps_hat, quant_axis):
    """
    Decompose a polarization vector into spherical components (eps_{+1}, eps_0, eps_{-1})
    relative to the quantization axis.

    Constructs a right-handed orthonormal frame {e1, e2, q_hat} from quant_axis,
    then projects eps_hat onto the spherical unit vectors:

        e_{+1} = -(e1 + i*e2) / sqrt(2)
        e_{-1} = +(e1 - i*e2) / sqrt(2)
        e_0    =  q_hat

    The spherical components are eps_q = eps_hat . e*_q (dot with conjugate):

        eps_{+1} = -(eps.e1 - i*eps.e2) / sqrt(2)    [drives Delta_m = +1, sigma+]
        eps_{-1} = +(eps.e1 + i*eps.e2) / sqrt(2)    [drives Delta_m = -1, sigma-]
        eps_0    =   eps.q_hat                         [drives Delta_m =  0, pi]

    Works for real (linear) and complex (circular/elliptical) polarization.

    Args:
        eps_hat    (array-like): Polarization unit vector, real or complex, shape (3,).
        quant_axis (array-like): Quantization axis unit vector [x, y, z], shape (3,).

    Returns:
        eps_p1 (complex): eps_{+1} component.
        eps_0  (complex): eps_0   component.
        eps_m1 (complex): eps_{-1} component.
    """
    q   = np.array(quant_axis, dtype=float)
    q   = q / np.linalg.norm(q)
    eps = np.array(eps_hat, dtype=complex)

    # Build orthonormal transverse frame; pick reference vector not parallel to q
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(q, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(q, ref);  e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(q, e1);   e2 = e2 / np.linalg.norm(e2)

    # Standard dot product (not Hermitian) since e1, e2 are real
    eps_p1 = -(np.dot(eps, e1) - 1j * np.dot(eps, e2)) / np.sqrt(2)
    eps_0  =   np.dot(eps, q)
    eps_m1 =  (np.dot(eps, e1) + 1j * np.dot(eps, e2)) / np.sqrt(2)

    return eps_p1, eps_0, eps_m1


def get_coupling_factor(eps_hat, quant_axis, mJ_target, cg_coeff=1.0):
    """
    Compute the effective polarization coupling factor for driving a specific
    Zeeman sublevel.

    The effective Rabi frequency for sublevel mJ is:
        Omega_eff = Omega_ref * |eps_q| * |CG|

    where Omega_ref = Gamma * sqrt(I / 2*I_sat) assumes maximal coupling
    (sigma+ cycling transition), q = mJ_target, and CG is the Clebsch-Gordan
    coefficient for the |g, mJ=0> -> |e, mJ_target> transition.

    For 1S0 (J=0) -> 3P1 (J=1), CG = 1 for all sublevels (default).

    Args:
        eps_hat    (array-like): Polarization unit vector, real or complex, shape (3,).
        quant_axis (array-like): Quantization axis unit vector [x, y, z].
        mJ_target  (int):        Target Zeeman sublevel: -1, 0, or +1.
        cg_coeff   (float):      Clebsch-Gordan coefficient. Default 1.0 (1S0->3P1).

    Returns:
        float: Coupling factor in [0, 1].
    """
    eps_p1, eps_0, eps_m1 = decompose_polarization(eps_hat, quant_axis)
    components = {+1: eps_p1, 0: eps_0, -1: eps_m1}
    return abs(components[mJ_target]) * abs(cg_coeff)


def get_zeeman_detuning(g_J, mJ, B_field):
    """
    Return the Zeeman frequency shift for a sublevel [rad/s].

        delta_Z = g_J * mu_B * B * mJ / hbar

    Args:
        g_J     (float): Lande g-factor.
        mJ      (int):   Magnetic quantum number.
        B_field (float): Bias magnetic field magnitude [T].

    Returns:
        float: Zeeman shift [rad/s]. Positive means the sublevel is shifted up in energy.
    """
    return g_J * MU_B * B_field * mJ / HBAR


def get_calculated_parameters(position, velocity, k_vecs, powers, beam_radii,
                               pol_vecs, quant_axis, mJ_targets):
    """
    Compute Doppler shifts and local Rabi frequencies for three beams,
    accounting for beam geometry, polarization, and Zeeman sublevel selection.

    Beams:
        beam_0: 689 nm  (Sr intercombination line, Gamma = 2pi x 7.48 kHz)
        beam_1: 688 nm  (Gamma = 2pi x 3.90 MHz)
        beam_2: 679 nm  (Gamma = 2pi x 1.26 MHz)

    Doppler shift:   Delta = -k_vec . v_vec
    Beam intensity:  I(r_perp) = (2P / pi w^2) exp(-2 r_perp^2 / w^2)  [uW/cm^2]
    Rabi frequency:  Omega = C * Gamma * sqrt(I / 2*I_sat)
                     where C = |eps_q| * |CG| is the polarization coupling factor.

    Args:
        position   (ndarray): Atom positions, shape (N, 3) [m].
        velocity   (ndarray): Atom velocities, shape (N, 3) [m/s].
        k_vecs     (tuple):   Wavevectors (k0, k1, k2), each shape (3,) [rad/m].
        powers     (tuple):   Beam powers (P0, P1, P2) [W].
        beam_radii (tuple):   1/e^2 beam radii (w0, w1, w2) [m].
        pol_vecs   (tuple):   Polarization unit vectors (eps0, eps1, eps2), each
                              shape (3,); real for linear, complex for circular.
        quant_axis (ndarray): Quantization axis unit vector [x, y, z], shape (3,).
        mJ_targets (tuple):   Target Zeeman sublevel for each beam (-1, 0, or +1).

    Returns:
        dict: Keys "beam_0", "beam_1", "beam_2", each containing:
              - "dshift" (ndarray): Doppler shift [rad/s], shape (N,).
              - "Omega"  (ndarray): Rabi frequency [rad/s], shape (N,).
    """
    k_vec_0, k_vec_1, k_vec_2 = k_vecs
    P0, P1, P2 = powers
    w0, w1, w2 = beam_radii

    doppler_0 = -np.sum(k_vec_0 * velocity, axis=1)
    doppler_1 = -np.sum(k_vec_1 * velocity, axis=1)
    doppler_2 = -np.sum(k_vec_2 * velocity, axis=1)

    r_perp_0 = get_effective_r_perp(position, k_vec_0)
    r_perp_1 = get_effective_r_perp(position, k_vec_1)
    r_perp_2 = get_effective_r_perp(position, k_vec_2)

    I0 = (2*P0 / (np.pi * w0**2)) * np.exp(-2 * r_perp_0**2 / w0**2) * 100  # [uW/cm^2]
    I1 = (2*P1 / (np.pi * w1**2)) * np.exp(-2 * r_perp_1**2 / w1**2) * 100  # [uW/cm^2]
    I2 = (2*P2 / (np.pi * w2**2)) * np.exp(-2 * r_perp_2**2 / w2**2) * 100  # [uW/cm^2]

    # Polarization coupling factors (|eps_q| x |CG|); CG = 1 for 1S0->3P1 on all beams
    C0 = get_coupling_factor(pol_vecs[0], quant_axis, mJ_targets[0])
    C1 = get_coupling_factor(pol_vecs[1], quant_axis, mJ_targets[1])
    C2 = get_coupling_factor(pol_vecs[2], quant_axis, mJ_targets[2])

    rabi_0 = C0 * gamma_689 * np.sqrt(I0 / (2 * Is_689))
    rabi_1 = C1 * gamma_688 * np.sqrt(I1 / (2 * Is_688))
    rabi_2 = C2 * gamma_679 * np.sqrt(I2 / (2 * Is_679))

    return {
        "beam_0": {"dshift": doppler_0, "Omega": rabi_0, "intensity": I0},
        "beam_1": {"dshift": doppler_1, "Omega": rabi_1, "intensity": I1},
        "beam_2": {"dshift": doppler_2, "Omega": rabi_2, "intensity": I2},
    }


def apply_readout(population, t_push):
    """
    Apply state-selective readout model to the excited-state population.

    After the Rabi pulse, a 461nm push beam of duration t_push selects
    ground-state (1S0) atoms. During t_push, excited-state (3P1) atoms
    can spontaneously decay to 1S0 and get pushed away, causing them to
    be miscounted as ground-state atoms. The survival probability is
    exp(-gamma_689 * t_push). Atoms that decay after the push remain in
    the unpushed cloud and are still correctly counted as |e>.

    The measured excited-state population is:
        P_meas = exp(-gamma_689 * t_push) * P_e

    Args:
        population (array-like): True excited-state population P_e.
        t_push     (float):      Duration of 461nm push pulse [s].

    Returns:
        ndarray: Measured excited-state population.
    """
    return np.exp(-gamma_689 * t_push) * np.asarray(population)


def simulate_one_photon_rabi_dynamics(positions, velocities, beam_radii,
                                      powers, detunings, k_vecs,
                                      pol_vecs, quant_axis, mJ_targets,
                                      t_max=5e-6, dt=10e-9,
                                      envelope='SQUARE',
                                      envelope_params=None,
                                      ensemble_params=None):
    """
    Simulate 689 nm single-photon Rabi dynamics for an atomic ensemble.

    Two operating modes, selected by `envelope`:

    SQUARE (default)
        The Hamiltonian is time-independent.  mesolve evolves the full
        tlist in one call per atom and returns population at every time
        step.  The provided positions/velocities are used as-is.

    ERF / GAUSSIAN / BLACKMAN  -- shot-by-shot scan
        Each element of tlist is treated as a pulse duration.  For each
        duration a complete shaped pulse is simulated independently and
        only the final excited-state population is recorded.  This
        matches a real experiment where you program a list of pulse
        times and measure the resulting population for each shot.

        If ensemble_params is given, atom positions and velocities are
        re-sampled from the thermal distribution for every shot so that
        each point on the curve uses an independent atomic ensemble.

    The RWA Hamiltonian (hbar = 1):
        H = Omega(t)/2 * sigma_x  -  Delta_eff/2 * sigma_z
    where Delta_eff = (-k.v) + detunings[0].  Spontaneous decay at
    Gamma_689 is included via a Lindblad collapse operator.

    Args:
        positions      (ndarray): Atom positions, shape (N, 3) [m].
                                  Used directly for SQUARE or when
                                  ensemble_params is None.
        velocities     (ndarray): Atom velocities, shape (N, 3) [m/s].
        beam_radii     (tuple):   1/e^2 beam radii (w0, w1, w2) [m].
        powers         (tuple):   Beam powers (P0, P1, P2) [W].
        detunings      (list):    Laser detunings [rad/s]; detunings[0]
                                  for the 689 nm beam.
        k_vecs         (tuple):   Wavevectors (k0, k1, k2) [rad/m].
        pol_vecs       (tuple):   Polarization unit vectors (eps0, eps1, eps2).
        quant_axis     (ndarray): Quantization axis unit vector, shape (3,).
        mJ_targets     (tuple):   Target Zeeman sublevel per beam.
        t_max          (float):   Max time or pulse duration [s].
        dt             (float):   Time step [s].
        envelope       (str):     'SQUARE', 'ERF', 'GAUSSIAN', or 'BLACKMAN'.
        envelope_params (dict or None):
                                  Shape-specific fixed parameters.
                                  ERF      : {'t0': float [s],
                                              'sigma': float [s]}
                                  GAUSSIAN : {'t0': float [s]}  (opt.)
                                  BLACKMAN : {'t0': float [s]}  (opt.)
                                  t0 defaults to 0.0 if omitted.
        ensemble_params (dict or None):
                                  If provided, atoms are re-sampled each
                                  shot (shaped-pulse mode only).
                                  Keys: 'radii', 'temperatures', 'n_atoms',
                                  and optionally 'mass' (default 88 amu).

    Returns:
        tlist          (ndarray): Time / pulse-duration axis [s].
        avg_population (ndarray): Ensemble-averaged excited-state population.
                                  SQUARE : population during continuous evolution.
                                  Shaped : final population after each shot.
    """
    n_steps = int(t_max / dt) + 1
    tlist   = np.linspace(0, t_max, n_steps)

    # Shared two-level operators
    g      = qt.basis(2, 0)
    sm     = qt.destroy(2)              # lowering operator  |g><e|
    sp     = qt.create(2)               # raising operator   |e><g|
    proj_e = sp * sm                    # excited-state projector |e><e|
    rho0   = g * g.dag()                # initial state: ground state
    c_ops  = [np.sqrt(gamma_689) * sm]  # spontaneous decay at Gamma_689

    # ------------------------------------------------------------------ #
    # SQUARE: constant-H mesolve, return full time trace                  #
    # ------------------------------------------------------------------ #
    if envelope == 'SQUARE':
        N_atoms        = positions.shape[0]
        avg_population = np.zeros(n_steps)
        params         = get_calculated_parameters(
            positions, velocities, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        for i in tqdm(range(N_atoms), desc='atoms (SQUARE)'):
            dshift = params['beam_0']['dshift'][i]
            rabi   = params['beam_0']['Omega'][i]
            delta  = dshift + detunings[0]
            H_sys  = (rabi * qt.sigmax() - delta * qt.sigmaz()) / 2
            result = qt.mesolve(H_sys, rho0, tlist, c_ops, [proj_e])
            avg_population += result.expect[0]

        return tlist, avg_population / N_atoms

    # ------------------------------------------------------------------ #
    # SHAPED: shot-by-shot scan over pulse durations                      #
    # ------------------------------------------------------------------ #
    if envelope not in ('ERF', 'GAUSSIAN', 'BLACKMAN'):
        raise ValueError(f"Unknown envelope '{envelope}'. "
                         "Choose from: SQUARE, ERF, GAUSSIAN, BLACKMAN.")

    ep = envelope_params or {}
    t0 = ep.get('t0', 0.0)

    avg_population = np.zeros(n_steps)

    for i, t_pulse in enumerate(tqdm(tlist, desc=f'shots ({envelope})')):
        

        # --- Atomic ensemble for this shot ---
        if ensemble_params is not None:
            pos_i, vel_i = sample_atomic_ensemble(
                radii=ensemble_params['radii'],
                temperatures=ensemble_params['temperatures'],
                mass=ensemble_params.get('mass', 88 * AMU),
                n_samples=ensemble_params['n_atoms'],
            )
        else:
            pos_i, vel_i = positions, velocities

        N_atoms_i = pos_i.shape[0]
        params_i  = get_calculated_parameters(
            pos_i, vel_i, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        if t_pulse == 0.0:
            avg_population[i] = 0.0
            continue

        # --- Simulation time window for this shot ---
        # ERF: extend 100 ns beyond each edge to capture erf rise/fall tails.
        # Others: pulse is exactly [t0, t0 + t_pulse].
        if envelope == 'ERF' or envelope == "GAUSSIAN":
            # add 150ns and 100 ns to start and end of 
            # simulation to account for rise and fall time of the pulse
            t_start = t0 - 150e-9
            t_end   = t0 + t_pulse + 100e-9
        else:
            t_start = t0
            t_end   = t0 + t_pulse

        n_sim     = max(2, int((t_end - t_start) / dt) + 1)
        tlist_sim = np.linspace(t_start, t_end, n_sim)

        shot_pop = 0.0
        for j in range(N_atoms_i):
            dshift  = params_i['beam_0']['dshift'][j]
            rabi_j  = params_i['beam_0']['Omega'][j]
            delta_j = dshift + detunings[0]

            # Build envelope scaled to this atom's peak Rabi frequency
            if envelope == 'ERF':
                
                coeff, _ = erf_rabi_envelope(
                    t0, ep.get('sigma', 90e-9), t_pulse, Omega_peak=rabi_j)
            elif envelope == 'GAUSSIAN':
                coeff, _ = gaussian_rabi_envelope(t0, t_pulse, Omega_peak=rabi_j)
            else:  # BLACKMAN
                coeff, _ = blackman_rabi_envelope(t0, t_pulse, Omega_peak=rabi_j)

            H_sys  = [-0.5 * delta_j * qt.sigmaz(), [0.5 * qt.sigmax(), coeff]]
            result = qt.mesolve(H_sys, rho0, tlist_sim, c_ops, [proj_e])
            shot_pop += result.expect[0][-1]  # final population after the pulse

        avg_population[i] = shot_pop / N_atoms_i

    return tlist, avg_population


def simulate_three_photon_rabi_dynamics(positions, velocities, beam_radii,
                                         powers, detunings, k_vecs,
                                         pol_vecs, quant_axis, mJ_targets,
                                         t_max=20e-6, dt=10e-9,
                                         n_shots=None,
                                         envelope='SQUARE',
                                         envelope_params=None,
                                         ensemble_params=None,
                                         n_jobs=None):
    """
    Simulate three-photon Rabi dynamics for the 1S0–3P1–3S1–3P0 ladder in Sr-88.

    Four-level system (basis index → state):
        0: 1S0   (ground state)
        1: 3P1   (first intermediate, typically mJ=+1)
        2: 3S1   (second intermediate)
        3: 3P0   (target state)

    Beams:
        beam_0 (689 nm): 1S0 → 3P1
        beam_1 (688 nm): 3P1 → 3S1
        beam_2 (679 nm): 3S1 → 3P0

    RWA Hamiltonian (ħ = 1):
        H_det  = −Δ₀|1⟩⟨1| − (Δ₀+Δ₁)|2⟩⟨2| − (Δ₀+Δ₁+Δ₂)|3⟩⟨3|
        H_drv  = Ω₀(t)/2 · (|0⟩⟨1|+h.c.)
               + Ω₁(t)/2 · (|1⟩⟨2|+h.c.)
               + Ω₂(t)/2 · (|2⟩⟨3|+h.c.)

    where Δᵢ = detunings[i] + Doppler_i (Doppler shift = −kᵢ·v).
    The cumulative rotating-frame energies are:
        level 1: −Δ₀
        level 2: −(Δ₀+Δ₁)
        level 3: −(Δ₀+Δ₁+Δ₂)

    Spontaneous decay Lindblad operators:
        √Γ₆₈₉ · |0⟩⟨1|     3P1 → 1S0
        √Γ₆₈₈ · |1⟩⟨2|     3S1 → 3P1  (partial A-coefficient)
        √Γ₆₇₉ · |3⟩⟨2|     3S1 → 3P0  (partial A-coefficient)

    Note: the 3S1→3P2 channel (~707 nm) is not modelled; remaining 3S1
    population that would decay that way is treated as not decaying.

    Modes:
        SQUARE          — constant H, returns full time trace of P(3P0).
        ERF / GAUSSIAN / BLACKMAN — shot-by-shot scan; each tlist element is
                          a pulse duration; returns P(3P0) after each shot.

    Args:
        positions      (ndarray): Atom positions, shape (N, 3) [m].
        velocities     (ndarray): Atom velocities, shape (N, 3) [m/s].
        beam_radii     (tuple):   1/e^2 beam radii (w0, w1, w2) [m].
        powers         (tuple):   Beam powers (P0, P1, P2) [W].
        detunings      (list):    Residual laser detunings [rad/s]; detunings[i]
                                  is the detuning of beam i from its transition
                                  (after accounting for Zeeman tuning).
        k_vecs         (tuple):   Wavevectors (k0, k1, k2) [rad/m].
        pol_vecs       (tuple):   Polarization unit vectors (eps0, eps1, eps2).
        quant_axis     (ndarray): Quantization axis unit vector, shape (3,).
        mJ_targets     (tuple):   Polarization selection (delta_mJ) per beam:
                                  +1 = sigma+, 0 = pi, -1 = sigma−.
        t_max          (float):   Max time or pulse duration [s].
        dt             (float):   Time step [s].
        envelope       (str):     'SQUARE', 'ERF', 'GAUSSIAN', or 'BLACKMAN'.
        envelope_params (dict or None): Same as one-photon version.
        ensemble_params (dict or None): Same as one-photon version.

    Returns:
        tlist           (ndarray): Time / pulse-duration axis [s].
        avg_populations (ndarray): Ensemble-averaged state populations,
                                   shape (4, n_steps). Rows correspond to
                                   1S0 (0), 3P1 (1), 3P0 (2), 3P2 (3).
                                   3S1 is not tracked (decays on ns timescales).
    """
    # For shaped modes, n_shots sets how many pulse durations to simulate (the
    # "experimental shots").  dt still controls the internal mesolve resolution.
    # For SQUARE mode n_shots is ignored — dt governs the output time grid.
    if n_shots is not None and envelope != 'SQUARE':
        n_steps = n_shots
    else:
        n_steps = int(t_max / dt) + 1
    tlist = np.linspace(0, t_max, n_steps)

    # 5-level basis: b[4] = 3P2 is a metastable sink (no H couplings, no outgoing decay)
    dim  = 5
    b    = [qt.basis(dim, i) for i in range(dim)]   # b[0]=1S0, b[1]=3P1, b[2]=3S1, b[3]=3P0, b[4]=3P2
    rho0 = b[0] * b[0].dag()                         # atoms start in 1S0

    # Coupling operators (off-diagonal blocks, symmetric; 3P2 is never driven)
    H_01 = 0.5 * (b[0] * b[1].dag() + b[1] * b[0].dag())   # 1S0 ↔ 3P1
    H_12 = 0.5 * (b[1] * b[2].dag() + b[2] * b[1].dag())   # 3P1 ↔ 3S1
    H_23 = 0.5 * (b[2] * b[3].dag() + b[3] * b[2].dag())   # 3S1 ↔ 3P0

    # Diagonal projectors; track 1S0, 3P1, 3P0, 3P2 — skip 3S1 (fast intermediate)
    proj  = [b[i] * b[i].dag() for i in range(dim)]
    e_ops = [proj[0], proj[1], proj[3], proj[4]]   # rows: 1S0, 3P1, 3P0, 3P2

    # Spontaneous decay Lindblad operators
    c_ops = [
        np.sqrt(gamma_689)         * b[0] * b[1].dag(),   # 3P1 → 1S0
        np.sqrt(gamma_688)         * b[1] * b[2].dag(),   # 3S1 → 3P1
        np.sqrt(gamma_679)         * b[3] * b[2].dag(),   # 3S1 → 3P0
        np.sqrt(gamma_707)         * b[4] * b[2].dag(),   # 3S1 → 3P2 (metastable sink)
    ]

    # ------------------------------------------------------------------ #
    # SQUARE: constant-H mesolve, return full time trace                  #
    # ------------------------------------------------------------------ #
    if envelope == 'SQUARE':
        N_atoms         = positions.shape[0]
        avg_populations = np.zeros((4, n_steps))
        par             = get_calculated_parameters(
            positions, velocities, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        for i in range(N_atoms):
            d0   = par['beam_0']['dshift'][i] + detunings[0]
            d01  = d0  + par['beam_1']['dshift'][i] + detunings[1]
            d012 = d01 + par['beam_2']['dshift'][i] - detunings[2]
            O0   = par['beam_0']['Omega'][i]
            O1   = par['beam_1']['Omega'][i]
            O2   = par['beam_2']['Omega'][i]


            H_det  = -d0 * proj[1] - d01 * proj[2] - d012 * proj[3]
            H_sys  = H_det + O0 * H_01 + O1 * H_12 + O2 * H_23

            result = qt.mesolve(H_sys, rho0, tlist, c_ops, e_ops=e_ops)
            avg_populations += np.array(result.expect)   # shape (4, n_steps)

        return tlist, avg_populations / N_atoms
    # ------------------------------------------------------------------ #
    # SHAPED: shot-by-shot scan over pulse durations                      #
    # ------------------------------------------------------------------ #
    if envelope not in ('ERF', 'GAUSSIAN', 'BLACKMAN'):
        raise ValueError(f"Unknown envelope '{envelope}'. "
                         "Choose from: SQUARE, ERF, GAUSSIAN, BLACKMAN.")

    ep = envelope_params or {}
    t0 = ep.get('t0', 0.0)

    avg_populations = np.zeros((4, n_steps))

    for i, t_pulse in enumerate(tqdm(tlist, desc=f'shots ({envelope} 3γ)')):

        # --- Atomic ensemble for this shot ---
        if ensemble_params is not None:
            pos_i, vel_i = sample_atomic_ensemble(
                radii=ensemble_params['radii'],
                temperatures=ensemble_params['temperatures'],
                mass=ensemble_params.get('mass', 88 * AMU),
                n_samples=ensemble_params['n_atoms'],
            )
        else:
            pos_i, vel_i = positions, velocities

        N_atoms_i = pos_i.shape[0]
        par_i     = get_calculated_parameters(
            pos_i, vel_i, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        if t_pulse == 0.0:
            avg_populations[:, i] = 0.0
            continue

        # --- Simulation time window ---
        if envelope in ('ERF', 'GAUSSIAN'):
            t_start = t0 - 150e-9
            t_end   = t0 + t_pulse + 100e-9
        else:
            t_start = t0
            t_end   = t0 + t_pulse

        n_sim     = max(2, int((t_end - t_start) / dt) + 1)
        tlist_sim = np.linspace(t_start, t_end, n_sim)

        shot_pop = np.zeros(4)
        for j in range(N_atoms_i):
            d0   = par_i['beam_0']['dshift'][j] + detunings[0]
            d01  = d0  + par_i['beam_1']['dshift'][j] + detunings[1]
            d012 = d01 + par_i['beam_2']['dshift'][j] - detunings[2]
            O0   = par_i['beam_0']['Omega'][j]
            O1   = par_i['beam_1']['Omega'][j]
            O2   = par_i['beam_2']['Omega'][j]

            H_det = -d0 * proj[1] - d01 * proj[2] - d012 * proj[3]
            if i == len(tlist)-1:
                f, _ = erf_rabi_envelope(t0, sigma_ep, t_pulse, Omega_peak=1)
                plt.plot(tlist*1e6, f(tlist))
                plt.show()

            if envelope == 'ERF':
                sigma_ep = ep.get('sigma', 90e-9)
                c0, _ = erf_rabi_envelope(t0, sigma_ep, t_pulse, Omega_peak=O0)
                c1, _ = erf_rabi_envelope(t0, sigma_ep, t_pulse, Omega_peak=O1)
                c2, _ = erf_rabi_envelope(t0, sigma_ep, t_pulse, Omega_peak=O2)
            elif envelope == 'GAUSSIAN':
                c0, _ = gaussian_rabi_envelope(t0, t_pulse, Omega_peak=O0)
                c1, _ = gaussian_rabi_envelope(t0, t_pulse, Omega_peak=O1)
                c2, _ = gaussian_rabi_envelope(t0, t_pulse, Omega_peak=O2)
            else:  # BLACKMAN
                c0, _ = blackman_rabi_envelope(t0, t_pulse, Omega_peak=O0)
                c1, _ = blackman_rabi_envelope(t0, t_pulse, Omega_peak=O1)
                c2, _ = blackman_rabi_envelope(t0, t_pulse, Omega_peak=O2)

            H_sys  = [H_det, [H_01, c0], [H_12, c1], [H_23, c2]]
            result = qt.mesolve(H_sys, rho0, tlist_sim, c_ops, e_ops=e_ops)
            shot_pop += np.array([result.expect[k][-1] for k in range(4)])

        avg_populations[:, i] = shot_pop / N_atoms_i

    return tlist, avg_populations


def erf_rabi_envelope(t0, sigma, t_pulse, Omega_peak=1.0):
    """
    Returns a QuTiP-compatible coefficient function  f(t, args)  that gives
    the Rabi frequency envelope shaped by the measured AOM rise and fall.

    The intensity profile is two back-to-back erf steps with an asymmetric
    fall (empirically measured from AOM data):

        t_on  = t0                        # 50% rise point
        t_off = t0 + t_pulse - 55e-9     # 50% fall point (AOM turns off 55 ns
                                          # before the RF pulse ends — empirical)

    Rabi frequency:
        Omega(t) = Omega_peak * sqrt( max(I(t), 0) )
    where
        I(t) = 0.5*(1 + erf((t - t_on ) / sigma))          <- rise
             * 0.5*(1 - erf((t - t_off) / (sigma * 0.65))) <- fall
                                          # fall is faster: sigma_fall = sigma * 0.65 (empirical)

    Parameters
    ----------
    t0 : float
        Time of the 50% intensity point on the rising edge [seconds].
    sigma : float
        Erf width for the rising edge [seconds].  The falling edge uses
        sigma * 0.65 (empirically measured to be faster than the rise).
    t_pulse : float
        Programmed RF pulse duration [seconds].  The 50% fall point is at
        t0 + t_pulse - 55 ns (empirical AOM timing offset).
    Omega_peak : float
        Peak Rabi frequency [rad/s].  Default 1.0 (returns normalised shape).

    Returns
    -------
    H_coeff : callable  f(t, args) -> float
        Drop this directly into a QuTiP time-dependent term:
            H = [H0, [H1, erf_rabi_envelope(...)]]
        args dict is unused but kept for QuTiP API compatibility.
    params : dict
        Dictionary of all parameters (handy for bookkeeping / mesolve args).

    Notes
    -----
    For t_pulse < 50 ns the AOM does not open meaningfully (t_off is at or
    before t_on), so the function short-circuits and returns a zero callable
    to avoid a wasted mesolve call.

    Example
    -------
    import qutip as qt
    H0    = -0.5 * delta * qt.sigmaz()
    H1    =  0.5 * qt.sigmax()          # Omega(t) prefactor goes here
    coeff, _ = erf_rabi_envelope(t0=0.0, sigma=90e-9,
                                 t_pulse=500e-9, Omega_peak=2*np.pi*1e6)
    result = qt.mesolve([H0, [H1, coeff]], psi0, tlist, [], [])
    """
    t_off = t0 + t_pulse - 55e-9  # 50% fall point (55 ns is an empirical AOM timing offset)
    t_on = t0 + 50e-9
    
    def f(t, _args=None):
        rise = 0.5 * (1.0 + erf((t-t_on)    / sigma))
        fall = 0.5 * (1.0 - erf((t - t_off) / (sigma * 0.65)))
        intensity =  rise * fall
        return Omega_peak * np.sqrt(np.maximum(intensity, 0.0))


    params = dict(t0=t0, sigma=sigma, t_pulse=t_pulse, Omega_peak=Omega_peak)
    if t_pulse < 50e-9:  # conservative cutoff: below ~55 ns t_off <= t_on so formula drives nothing
        return lambda t, _args=None: 0.0, params
    else:
        return f, params

def gaussian_rabi_envelope(t0, t_pulse, Omega_peak=1.0):
    """
    Returns a QuTiP-compatible coefficient function f(t, args) giving a
    Gaussian Rabi frequency envelope.

        Omega(t) = Omega_peak * exp(-0.5 * ((t - center) / sigma)^2)

    where center = t0 + t_pulse/2 and sigma = t_pulse/4 (~95% of pulse
    energy falls within [t0, t0 + t_pulse]).

    Parameters
    ----------
    t0 : float
        Start time of the pulse window [s].
    t_pulse : float
        Pulse duration [s]. Controls the centre and width of the Gaussian.
    Omega_peak : float
        Peak Rabi frequency [rad/s]. Default 1.0 (returns normalised shape).

    Returns
    -------
    H_coeff : callable  f(t, args) -> float
        Drop directly into a QuTiP time-dependent term.
        args dict is unused but kept for QuTiP API compatibility.
    params : dict
        Dictionary of all parameters.
    """
    params = dict(t0=t0, t_pulse=t_pulse, Omega_peak=Omega_peak)
    if t_pulse == 0:
        return lambda t, _args=None: np.zeros_like(np.asarray(t, dtype=float)), params

    center = t0 + t_pulse / 2
    sigma  = t_pulse / 4

    def f(t, _args=None):
        return Omega_peak * np.exp(-0.5 * ((np.asarray(t) - center) / sigma)**2)

    params.update(center=center, sigma=sigma)
    return f, params


def blackman_rabi_envelope(t0, t_pulse, Omega_peak=1.0):
    """
    Returns a QuTiP-compatible coefficient function f(t, args) giving a
    Blackman-windowed Rabi frequency envelope.

        Omega(t) = Omega_peak * (0.42 - 0.5*cos(2pi*(t-t0)/T)
                                       + 0.08*cos(4pi*(t-t0)/T))
                   for t in [t0, t0 + t_pulse], else 0.

    The Blackman window is naturally 0 at both edges and 1 at the centre,
    making it useful when spectral sidelobe suppression matters (e.g. atom
    interferometry, narrow-line spectroscopy).

    Parameters
    ----------
    t0 : float
        Start time of the pulse [s].
    t_pulse : float
        Pulse duration [s].
    Omega_peak : float
        Peak Rabi frequency [rad/s]. Default 1.0 (returns normalised shape).

    Returns
    -------
    H_coeff : callable  f(t, args) -> float
        Drop directly into a QuTiP time-dependent term.
        args dict is unused but kept for QuTiP API compatibility.
    params : dict
        Dictionary of all parameters.
    """
    params = dict(t0=t0, t_pulse=t_pulse, Omega_peak=Omega_peak)
    if t_pulse == 0:
        return lambda t, _args=None: np.zeros_like(np.asarray(t, dtype=float)), params

    t_end = t0 + t_pulse

    def f(t, _args=None):
        t    = np.asarray(t)
        mask = (t >= t0) & (t <= t_end)
        phase = 2 * PI * (t - t0) / t_pulse
        return Omega_peak * np.where(
            mask,
            0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase),
            0.0,
        )

    return f, params


def test_envelopes(t, widths, envelope):
    """
    Plot Rabi frequency envelope shapes (not intensity) for a range of pulse durations.

    Supported envelopes:
        "SQUARE"    - ideal rectangular pulse.
        "ERF"       - AOM-realistic rise/fall (sigma = 90 ns); calls erf_rabi_envelope.
        "GAUSSIAN"  - Gaussian centered at w/2 with sigma = w/4; calls gaussian_rabi_envelope.
        "BLACKMAN"  - Blackman window over [0, w]; calls blackman_rabi_envelope.

    Args:
        t        (ndarray): Time array [s].
        widths   (ndarray): Pulse durations to sweep over [s].
        envelope (str):     One of the envelope names above.
    """
    envelope_fns = {
        "ERF":      lambda w: erf_rabi_envelope(0, 90e-9, w),
        "GAUSSIAN": lambda w: gaussian_rabi_envelope(0, w),
        "BLACKMAN": lambda w: blackman_rabi_envelope(0, w),
    }

    if envelope == "SQUARE":
        for w in widths:
            plt.plot(t * 1e6, (t > 0) * (t < w))
    elif envelope in envelope_fns:
        for w in widths:
            if w == 0:
                continue
            f, _ = envelope_fns[envelope](w)
            plt.plot(t * 1e6, f(t))
    else:
        raise ValueError(f"Unknown envelope '{envelope}'. "
                         "Choose from: SQUARE, ERF, GAUSSIAN, BLACKMAN.")

    plt.title(f"Rabi Envelope Shapes - Pulse Shape = {envelope}")
    plt.ylabel("Normalized Rabi Freq.")
    plt.xlabel(r"Time ($\mu s$)")
    plt.xlim(t[0]*1e6, t[-1]*1e6)
    plt.show()


# ============================================================
# NEW: Strategies A + B + C
#   A — numpy/scipy Liouvillian instead of QuTiP mesolve
#   B — all atoms batched into a single ODE per shot
#   C — joblib parallelism across shots (ERF/GAUSSIAN/BLACKMAN modes)
#
# All new names carry the _new suffix.  No existing code is modified.
# ============================================================


# ── Pre-built 5-level numpy matrices (computed once at import) ─────────── #
_d5  = 5
_d25 = _d5 * _d5
_I5  = np.eye(_d5, dtype=complex)


def _ket5_new(i):
    v = np.zeros(_d5, dtype=complex)
    v[i] = 1.0
    return v


# Projectors |i><i| and coupling operators |i><j|+h.c. (all 5×5 complex)
_PROJ5_NP  = np.array([np.outer(_ket5_new(i), _ket5_new(i).conj())
                        for i in range(_d5)])          # shape (5, 5, 5)
_H01_NP    = 0.5 * (np.outer(_ket5_new(0), _ket5_new(1).conj()) +
                     np.outer(_ket5_new(1), _ket5_new(0).conj()))
_H12_NP    = 0.5 * (np.outer(_ket5_new(1), _ket5_new(2).conj()) +
                     np.outer(_ket5_new(2), _ket5_new(1).conj()))
_H23_NP    = 0.5 * (np.outer(_ket5_new(2), _ket5_new(3).conj()) +
                     np.outer(_ket5_new(3), _ket5_new(2).conj()))

# Collapse operators: same decay channels as the QuTiP version
_COPS5_NP  = [
    np.sqrt(gamma_689) * np.outer(_ket5_new(0), _ket5_new(1).conj()),  # 3P1→1S0
    np.sqrt(gamma_688) * np.outer(_ket5_new(1), _ket5_new(2).conj()),  # 3S1→3P1
    np.sqrt(gamma_679) * np.outer(_ket5_new(3), _ket5_new(2).conj()),  # 3S1→3P0
    np.sqrt(gamma_707) * np.outer(_ket5_new(4), _ket5_new(2).conj()),  # 3S1→3P2
]


def build_liouvillian_numpy(H_np, c_ops_list):
    """
    Build the d²×d² Liouvillian superoperator.

    Uses row-major vectorization: d/dt vec(ρ) = L @ vec(ρ),
    where vec(ρ) = ρ.flatten() (C order).

    Row-major Kronecker identities:
        vec(AρB)    = (A ⊗ B^T) vec(ρ)
        L_coherent  = −i (H ⊗ I − I ⊗ H^T)
        L_jump_k    =  C_k ⊗ C_k* − ½(C_k†C_k ⊗ I) − ½(I ⊗ (C_k†C_k)^T)

    Populations are on the diagonal: ρ[k,k] = vec(ρ)[k*d + k].

    Args:
        H_np       (ndarray): Hamiltonian, shape (d, d), complex.
        c_ops_list (list):    Collapse operators, each shape (d, d), complex.

    Returns:
        L (ndarray): Liouvillian, shape (d², d²), complex.
    """
    d = H_np.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H_np, I) - np.kron(I, H_np.T))
    for c in c_ops_list:
        cdc = c.conj().T @ c
        L += np.kron(c, c.conj()) - 0.5 * np.kron(cdc, I) - 0.5 * np.kron(I, cdc.T)
    return L


# Coherent Liouvillian for a single operator (no c_ops)
def _coh_L_new(H):
    d = H.shape[0]; I = np.eye(d, dtype=complex)
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


# Pre-built Liouvillian blocks — computed once at import time
_L_DISS_NP   = np.zeros((_d25, _d25), dtype=complex)
for _c in _COPS5_NP:
    _cdc = _c.conj().T @ _c
    _L_DISS_NP += (np.kron(_c, _c.conj())
                   - 0.5 * np.kron(_cdc, _I5)
                   - 0.5 * np.kron(_I5, _cdc.T))

_L_H01_NP    = _coh_L_new(_H01_NP)                             # (25, 25)
_L_H12_NP    = _coh_L_new(_H12_NP)
_L_H23_NP    = _coh_L_new(_H23_NP)
_L_PROJ5_NP  = np.array([_coh_L_new(_PROJ5_NP[i])
                          for i in range(_d5)])                 # (5, 25, 25)

# Initial density-matrix vector
_RHO0_VEC_NP = np.zeros(_d25, dtype=complex)

# _RHO0_VEC_NP[0] = 1.0  # initial pop in 1s0
# _RHO0_VEC_NP[6] = 1.0  # initial pop in 3p1
_RHO0_VEC_NP[18] = 1.0  # initial pop in 3p0


def _extract_pops_new(rho_flat_batch, N_atoms):
    """
    Extract ensemble-averaged populations for states 0,1,3,4 from a
    flattened batch density-matrix vector of shape (N_atoms * d25,).

    Returns ndarray of shape (4,).
    """
    d  = _d5
    d2 = _d25
    pop = np.zeros(4)
    for row, k in enumerate([0, 1, 3, 4]):
        idxs = np.arange(N_atoms) * d2 + k * d + k
        pop[row] = np.real(rho_flat_batch[idxs]).mean()
    return pop


def _build_L_batches_new(d0_arr, d01_arr, d012_arr, O0_arr, O1_arr, O2_arr):
    """
    Vectorised construction of per-atom Liouvillians.

    Exploits linearity of L in H:
        L_i = L_diss
              − d0_i·L(P₁) − d01_i·L(P₂) − d012_i·L(P₃)
              + O0_i·L(H₀₁) + O1_i·L(H₁₂) + O2_i·L(H₂₃)

    All Kronecker products are precomputed at module level; this function
    only does broadcasting + addition of (N, 25, 25) arrays.

    Returns
        L_static (N, 25, 25): detuning + decay part (time-independent)
        L_td     (N, 25, 25): coupling part (scaled by shape function in ERF mode)
    """
    sh = (len(d0_arr), 1, 1)
    L_static = (
        _L_DISS_NP[np.newaxis]
        - d0_arr.reshape(sh)   * _L_PROJ5_NP[1][np.newaxis]
        - d01_arr.reshape(sh)  * _L_PROJ5_NP[2][np.newaxis]
        - d012_arr.reshape(sh) * _L_PROJ5_NP[3][np.newaxis]
    )
    L_td = (
        O0_arr.reshape(sh) * _L_H01_NP[np.newaxis]
        + O1_arr.reshape(sh) * _L_H12_NP[np.newaxis]
        + O2_arr.reshape(sh) * _L_H23_NP[np.newaxis]
    )
    return L_static, L_td


def _make_shape_fn_new(envelope, t0_ep, t_pulse, ep):
    """Return the normalized scalar shape function s(t) ∈ [0,1] for one shot."""
    if envelope == 'ERF':
        sigma = ep.get('sigma', 90e-9)
        t_on  = t0_ep + 50e-9
        t_off = t0_ep + t_pulse - 55e-9
        def shape(t):
            rise = 0.5 * (1.0 + erf((t - t_on)  /  sigma))
            fall = 0.5 * (1.0 - erf((t - t_off) / (sigma * 0.65)))
            return np.sqrt(np.maximum(rise * fall, 0.0))
    elif envelope == 'GAUSSIAN':
        center = t0_ep + t_pulse / 2
        sig_g  = t_pulse / 4
        def shape(t):
            return np.exp(-0.5 * ((np.asarray(t) - center) / sig_g) ** 2)
    else:  # BLACKMAN
        t_end = t0_ep + t_pulse
        def shape(t):
            t   = np.asarray(t)
            msk = (t >= t0_ep) & (t <= t_end)
            ph  = 2 * PI * (t - t0_ep) / t_pulse
            return np.where(msk, 0.42 - 0.5 * np.cos(ph) + 0.08 * np.cos(2 * ph), 0.0)
    return shape


def _run_one_shot_new(t_pulse, L_static_b, L_td_b, rho0_b, N_s,
                       envelope, t0_ep, ep):
    """
    Integrate one shaped-pulse shot for all N_s atoms simultaneously (Strategy B).

    The batched ODE state has shape (N_s * 25,).  L_static_b and L_td_b
    have shape (N_s, 25, 25).  The shape function s(t) is a scalar evaluated
    once per ODE step, so the cost is dominated by two (N_s, 25, 25) @ (N_s, 25)
    batched matmuls per step — implemented as np.matmul on 3-D arrays.

    Returns pop : ndarray shape (4,) — ensemble-averaged populations for
                  states 0 (1S0), 1 (3P1), 3 (3P0), 4 (3P2).
    """
    if t_pulse == 0.0:
        return np.zeros(4)
    # ERF short-circuit: AOM doesn't open for < 50 ns (mirrors erf_rabi_envelope)
    if envelope == 'ERF' and t_pulse < 50e-9:
        return np.zeros(4)

    shape = _make_shape_fn_new(envelope, t0_ep, t_pulse, ep)

    if envelope in ('ERF', 'GAUSSIAN'):
        t_start = t0_ep - 150e-9
        t_end_s = t0_ep + t_pulse + 100e-9
    else:
        t_start = t0_ep
        t_end_s = t0_ep + t_pulse

    d2 = _d25

    def rhs(t, state):
        rho  = state.reshape(N_s, d2)
        s    = float(np.real(shape(t)))
        # batched matmul: (N_s,25,25)@(N_s,25,1) → (N_s,25)
        drho = np.matmul(L_static_b, rho[:, :, np.newaxis]).squeeze(-1)
        drho = drho + s * np.matmul(L_td_b, rho[:, :, np.newaxis]).squeeze(-1)
        return drho.flatten()

    sol = _solve_ivp_new(
        rhs,
        [t_start, t_end_s],
        rho0_b,
        method='RK45',
        rtol=1e-6, atol=1e-8,
        dense_output=False,
    )
    return _extract_pops_new(sol.y[:, -1], N_s)


def simulate_three_photon_rabi_dynamics_new(
        positions, velocities, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets,
        t_max=20e-6, dt=10e-9, n_shots=None,
        envelope='SQUARE', envelope_params=None, ensemble_params=None,
        n_jobs=1):
    """
    Fast drop-in replacement for simulate_three_photon_rabi_dynamics.

    Implements Strategies A, B, C from planning.md:
      A — numpy/scipy Liouvillian superoperator instead of QuTiP mesolve
      B — all N atoms batched into one ODE per shot (vectorised matmul RHS)
      C — shots parallelised with joblib (ERF/GAUSSIAN/BLACKMAN modes only)

    Extra parameter vs. original:
        n_jobs (int): joblib workers for shot-level parallelism.
                      1 = sequential (safe default).
                     -1 = use all logical CPU cores.

    Returns same (tlist, avg_populations) shape as the original.
    """
    d  = _d5
    d2 = _d25

    if n_shots is not None and envelope != 'SQUARE':
        n_steps = n_shots
    else:
        n_steps = int(t_max / dt) + 1
    tlist = np.linspace(0, t_max, n_steps)

    # ── SQUARE mode: constant H per atom — single batched ODE ─────────── #
    if envelope == 'SQUARE':
        N_atoms = positions.shape[0]
        par = get_calculated_parameters(
            positions, velocities, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        d0_arr   = par['beam_0']['dshift'] + detunings[0]
        d01_arr  = d0_arr  + par['beam_1']['dshift'] + detunings[1]
        d012_arr = d01_arr + par['beam_2']['dshift'] - detunings[2]
        O0_arr   = par['beam_0']['Omega']
        O1_arr   = par['beam_1']['Omega']
        O2_arr   = par['beam_2']['Omega']

        # SQUARE: static + td parts are always summed (s=1 always)
        L_static, L_td = _build_L_batches_new(
            d0_arr, d01_arr, d012_arr, O0_arr, O1_arr, O2_arr)
        L_batch = L_static + L_td   # (N_atoms, 25, 25), fully constant

        rho0_batch = np.tile(_RHO0_VEC_NP, N_atoms)  # (N_atoms*25,)

        def _rhs_square(t, state):
            rho  = state.reshape(N_atoms, d2)
            drho = np.matmul(L_batch, rho[:, :, np.newaxis]).squeeze(-1)
            return drho.flatten()

        sol = _solve_ivp_new(
            _rhs_square,
            [tlist[0], tlist[-1]],
            rho0_batch,
            method='RK45',
            t_eval=tlist,
            rtol=1e-8, atol=1e-10,
            dense_output=False,
        )

        avg_pops = np.zeros((4, n_steps))
        for row, k in enumerate([0, 1, 3, 4]):
            idxs = np.arange(N_atoms) * d2 + k * d + k
            avg_pops[row] = np.real(sol.y[idxs, :]).mean(axis=0)
        
        param_dict = {
            'beam_0': {
                'laser_detuning':    detunings[0],
                'mean_doppler':      float(np.mean(par['beam_0']['dshift'])),
                'std_doppler':       float(np.std(par['beam_0']['dshift'])),
                'mean_eff_detuning': float(np.mean(d0_arr)),
                'std_eff_detuning':  float(np.std(d0_arr)),
                'mean_Omega':        float(np.mean(O0_arr)),
                'std_Omega':         float(np.std(O0_arr)),
                'mean_intensity':    float(np.mean(par['beam_0']['intensity'])),
                'std_intensity':     float(np.std(par['beam_0']['intensity'])),
            },
            'beam_1': {
                'laser_detuning':    detunings[1],
                'mean_doppler':      float(np.mean(par['beam_1']['dshift'])),
                'std_doppler':       float(np.std(par['beam_1']['dshift'])),
                'mean_eff_detuning': float(np.mean(d01_arr)),
                'std_eff_detuning':  float(np.std(d01_arr)),
                'mean_Omega':        float(np.mean(O1_arr)),
                'std_Omega':         float(np.std(O1_arr)),
                'mean_intensity':    float(np.mean(par['beam_1']['intensity'])),
                'std_intensity':     float(np.std(par['beam_1']['intensity'])),
            },
            'beam_2': {
                'laser_detuning':    detunings[2],
                'mean_doppler':      float(np.mean(par['beam_2']['dshift'])),
                'std_doppler':       float(np.std(par['beam_2']['dshift'])),
                'mean_eff_detuning': float(np.mean(d012_arr)),
                'std_eff_detuning':  float(np.std(d012_arr)),
                'mean_Omega':        float(np.mean(O2_arr)),
                'std_Omega':         float(np.std(O2_arr)),
                'mean_intensity':    float(np.mean(par['beam_2']['intensity'])),
                'std_intensity':     float(np.std(par['beam_2']['intensity'])),
            },
        }
        return tlist, avg_pops, param_dict

    # ── SHAPED (ERF / GAUSSIAN / BLACKMAN) mode ───────────────────────── #
    if envelope not in ('ERF', 'GAUSSIAN', 'BLACKMAN'):
        raise ValueError(f"Unknown envelope '{envelope}'. "
                         "Choose from: SQUARE, ERF, GAUSSIAN, BLACKMAN.")

    ep    = envelope_params or {}
    t0_ep = ep.get('t0', 0.0)

    # Build per-shot atomic ensembles and their Liouvillian batches
    shot_matrices = []
    # accumulate per-atom values across shots for diagnostics
    all_dop0, all_dop1, all_dop2 = [], [], []
    all_d0,   all_d01,  all_d012 = [], [], []
    all_O0,   all_O1,   all_O2   = [], [], []
    all_I0,   all_I1,   all_I2   = [], [], []

    for idx in range(n_steps):
        if ensemble_params is not None:
            pos_i, vel_i = sample_atomic_ensemble(
                radii=ensemble_params['radii'],
                temperatures=ensemble_params['temperatures'],
                mass=ensemble_params.get('mass', 88 * AMU),
                n_samples=ensemble_params['n_atoms'],
            )
            pos_i = np.atleast_2d(pos_i)
            vel_i = np.atleast_2d(vel_i)
        else:
            pos_i = np.atleast_2d(positions)
            vel_i = np.atleast_2d(velocities)

        N_s = pos_i.shape[0]
        par_i = get_calculated_parameters(
            pos_i, vel_i, k_vecs, powers,
            beam_radii, pol_vecs, quant_axis, mJ_targets)

        d0_s   = par_i['beam_0']['dshift'] + detunings[0]
        d01_s  = d0_s  + par_i['beam_1']['dshift'] + detunings[1]
        d012_s = d01_s + par_i['beam_2']['dshift'] - detunings[2]
        O0_s   = par_i['beam_0']['Omega']
        O1_s   = par_i['beam_1']['Omega']
        O2_s   = par_i['beam_2']['Omega']

        all_dop0.append(par_i['beam_0']['dshift']); all_dop1.append(par_i['beam_1']['dshift']); all_dop2.append(par_i['beam_2']['dshift'])
        all_d0.append(d0_s);   all_d01.append(d01_s);   all_d012.append(d012_s)
        all_O0.append(O0_s);   all_O1.append(O1_s);     all_O2.append(O2_s)
        all_I0.append(par_i['beam_0']['intensity']); all_I1.append(par_i['beam_1']['intensity']); all_I2.append(par_i['beam_2']['intensity'])

        L_stat, L_td = _build_L_batches_new(
            d0_s, d01_s, d012_s, O0_s, O1_s, O2_s)
        rho0_b = np.tile(_RHO0_VEC_NP, N_s)
        shot_matrices.append((L_stat, L_td, rho0_b, N_s))

    # parallelise shots with joblib
    # prefer="threads" avoids broken-process-pool errors in interactive
    # environments (VS Code, Jupyter) on Windows; numpy/scipy release the GIL
    # so thread-based parallelism still gives real speedup.
    pop_list = _Parallel_new(n_jobs=n_jobs, prefer="threads")(
        _delayed_new(_run_one_shot_new)(
            t_pulse, L_stat, L_td, rho0_b, N_s, envelope, t0_ep, ep)
        for t_pulse, (L_stat, L_td, rho0_b, N_s)
        in zip(tlist, shot_matrices)
    )

    avg_populations = np.array(pop_list).T   # (4, n_steps)

    # flatten all per-atom values across shots for aggregate stats
    cat0   = np.concatenate(all_dop0); cat1   = np.concatenate(all_dop1); cat2   = np.concatenate(all_dop2)
    catd0  = np.concatenate(all_d0);   catd01 = np.concatenate(all_d01);  catd012 = np.concatenate(all_d012)
    catO0  = np.concatenate(all_O0);   catO1  = np.concatenate(all_O1);   catO2  = np.concatenate(all_O2)
    catI0  = np.concatenate(all_I0);   catI1  = np.concatenate(all_I1);   catI2  = np.concatenate(all_I2)

    param_dict = {
        'beam_0': {
            'laser_detuning':    detunings[0],
            'mean_doppler':      float(np.mean(cat0)),
            'std_doppler':       float(np.std(cat0)),
            'mean_eff_detuning': float(np.mean(catd0)),
            'std_eff_detuning':  float(np.std(catd0)),
            'mean_Omega':        float(np.mean(catO0)),
            'std_Omega':         float(np.std(catO0)),
            'mean_intensity':    float(np.mean(catI0)),
            'std_intensity':     float(np.std(catI0)),
        },
        'beam_1': {
            'laser_detuning':    detunings[1],
            'mean_doppler':      float(np.mean(cat1)),
            'std_doppler':       float(np.std(cat1)),
            'mean_eff_detuning': float(np.mean(catd01)),
            'std_eff_detuning':  float(np.std(catd01)),
            'mean_Omega':        float(np.mean(catO1)),
            'std_Omega':         float(np.std(catO1)),
            'mean_intensity':    float(np.mean(catI1)),
            'std_intensity':     float(np.std(catI1)),
        },
        'beam_2': {
            'laser_detuning':    detunings[2],
            'mean_doppler':      float(np.mean(cat2)),
            'std_doppler':       float(np.std(cat2)),
            'mean_eff_detuning': float(np.mean(catd012)),
            'std_eff_detuning':  float(np.std(catd012)),
            'mean_Omega':        float(np.mean(catO2)),
            'std_Omega':         float(np.std(catO2)),
            'mean_intensity':    float(np.mean(catI2)),
            'std_intensity':     float(np.std(catI2)),
        },
    }
    return tlist, avg_populations, param_dict


def print_simulation_params(param_dict):
    """
    Print a formatted diagnostic summary of a param_dict returned by
    simulate_three_photon_rabi_dynamics_new.

    Args:
        param_dict (dict): As returned by simulate_three_photon_rabi_dynamics_new.
    """
    beam_labels  = ['beam_0 (689 nm)', 'beam_1 (688 nm)', 'beam_2 (679 nm)']
    beam_keys    = ['beam_0', 'beam_1', 'beam_2']
    detun_labels = ['Δ₀ (single-photon)', 'Δ₀₁ (two-photon)', 'Δ₀₁₂ (three-photon)']

    print("=" * 60)
    print("  Three-Photon Simulation — Parameter Summary")
    print("=" * 60)

    for label, key in zip(beam_labels, beam_keys):
        b = param_dict[key]
        print(f"\n  {label}")
        print(f"  {'─' * (len(label) + 2)}")
        print(f"    Laser detuning   : {b['laser_detuning'] / (2*PI) * 1e-6:+.3f} MHz")
        print(f"    Doppler shift    : {b['mean_doppler']   / (2*PI) * 1e-6:+.3f} ± "
              f"{b['std_doppler']   / (2*PI) * 1e-6:.3f} MHz")
        print(f"    Eff. detuning    : {b['mean_eff_detuning'] / (2*PI) * 1e-6:+.3f} ± "
              f"{b['std_eff_detuning'] / (2*PI) * 1e-6:.3f} MHz")
        print(f"    Rabi frequency Ω : {b['mean_Omega'] / (2*PI) * 1e-3:,.1f} ± "
              f"{b['std_Omega'] / (2*PI) * 1e-3:.1f} kHz")
        print(f"    Local intensity  : {b['mean_intensity']:,.2f} ± "
              f"{b['std_intensity']:.2f} µW/cm²")

    print("\n  Cumulative detunings (mean ± std across ensemble)")
    print(f"  {'─' * 50}")
    for dlabel, bkey in zip(detun_labels, beam_keys):
        b = param_dict[bkey]
        print(f"    {dlabel:<28}: {b['mean_eff_detuning'] / (2*PI) * 1e-6:+.3f} ± "
              f"{b['std_eff_detuning'] / (2*PI) * 1e-6:.3f} MHz")

    print("=" * 60)


