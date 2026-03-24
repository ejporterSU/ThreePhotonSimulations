#%%
##### IMPORTS ###############

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from tqdm import tqdm


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
        return positions[0], velocities[0]
    return positions, velocities


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
        "beam_0": {"dshift": doppler_0, "Omega": rabi_0},
        "beam_1": {"dshift": doppler_1, "Omega": rabi_1},
        "beam_2": {"dshift": doppler_2, "Omega": rabi_2},
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
                                      t_max=20e-6, dt=10e-9):
    """
    Simulate 689 nm single-photon Rabi dynamics for an atomic ensemble.

    Solves the Lindblad master equation independently for each atom using
    its local Rabi frequency and Doppler-shifted detuning, then returns
    the ensemble-averaged excited state population.

    The RWA Hamiltonian (in units where hbar = 1) is:
        H = (Omega * sigma_x - Delta_eff * sigma_z) / 2
    where Delta_eff = (-k.v) + detunings[0] is the total detuning from the
    target Zeeman sublevel (detunings[0] = 0 means laser is on resonance).
    The Rabi frequency Omega already includes the polarization coupling factor
    |eps_q| for the target mJ sublevel. Spontaneous decay at rate Gamma_689
    is included via a Lindblad collapse operator.

    Args:
        positions  (ndarray): Atom positions, shape (N, 3) [m].
        velocities (ndarray): Atom velocities, shape (N, 3) [m/s].
        beam_radii (tuple):   1/e^2 beam radii (w0, w1, w2) [m].
        powers     (tuple):   Beam powers (P0, P1, P2) [W].
        detunings  (list):    Laser detunings from target Zeeman sublevel [rad/s];
                              detunings[0] is used for the 689 nm beam.
        k_vecs     (tuple):   Wavevectors (k0, k1, k2), each shape (3,) [rad/m].
        pol_vecs   (tuple):   Polarization unit vectors (eps0, eps1, eps2), shape (3,).
        quant_axis (ndarray): Quantization axis unit vector, shape (3,).
        mJ_targets (tuple):   Target Zeeman sublevel for each beam (-1, 0, or +1).
        t_max      (float):   Total simulation time [s]. Default 20 us.
        dt         (float):   Time step [s]. Default 10 ns.

    Returns:
        tlist          (ndarray): Time points, shape (n_steps,) [s].
        avg_population (ndarray): Ensemble-averaged excited state population,
                                  shape (n_steps,).
    """
    N_atoms = positions.shape[0]
    n_steps = int(t_max / dt) + 1
    tlist   = np.linspace(0, t_max, n_steps)

    # Two-level system operators
    g      = qt.basis(2, 0)
    sm     = qt.destroy(2)              # lowering operator |g><e|
    sp     = qt.create(2)               # raising operator  |e><g|
    proj_e = sp * sm                    # excited state projector |e><e|

    rho0  = g * g.dag()                 # initial state: all atoms in ground state
    c_ops = [np.sqrt(gamma_689) * sm]   # spontaneous decay at rate Gamma_689

    avg_population = np.zeros(n_steps)
    params = get_calculated_parameters(positions, velocities, k_vecs, powers,
                                       beam_radii, pol_vecs, quant_axis, mJ_targets)

    for i in range(N_atoms):
        dshift = params['beam_0']['dshift'][i]
        rabi   = params['beam_0']['Omega'][i]
        delta  = dshift + detunings[0]

        H_sys  = (rabi * qt.sigmax() - delta * qt.sigmaz()) / 2
        result = qt.mesolve(H_sys, rho0, tlist, c_ops, [proj_e])
        avg_population += result.expect[0]

    return tlist, avg_population / N_atoms


if __name__ == "__main__":

    # raw data on res
    t_raw_us= np.array([0.        , 0.02727273, 0.05454545, 0.08181818, 0.10909091,
       0.13636364, 0.16363636, 0.19090909, 0.21818182, 0.24545455,
       0.27272727, 0.3       ]),

    pop_avg_raw= np.array([0.007218251193264217, 0.07522068434528818, 0.27351518028631533, 
        0.5334829295277013, 0.801381491260522, 0.9358372210313958, 
        0.8418133567975138, 0.6128245196916823, 0.31440830735719383, 
        0.13100533413670276, 0.03446727696389848, 0.1397543101734671])
    
    # raw data 5MHz detuned
    # t_raw_us= np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
    #    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
    #    0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 ])

    # pop_avg_raw=np.array([0.01215464554673015, 0.014655686947439102, 0.037922786309883084, 
    #                     0.08492874990406181, 0.12538398157733505, 0.14875453371038413,
    #                     0.19171904506891402, 0.21102177615794582, 0.21197614738259493, 
    #                     0.20019653549065314, 0.18369942762095787, 0.15096029843253678, 
    #                     0.09824514449986849, 0.06309188581263286, 0.026059112639586417, 
    #                     0.01183442594940693, 0.010750569651698676, 0.022536023299094064, 
    #                     0.03210180443526232, 0.08650610451372473, 0.1322141383429254, 
    #                     0.15385858977778855, 0.18613193369262693, 0.21471766947020343, 
    #                     0.22621477782233076, 0.17064615681781267, 0.1903908150467581, 
    #                     0.186064703817872, 0.15758128982514602, 0.1019319389542577, 
    #                     0.0767776308937227])






    w0, w1, w2       = 0.5e-3, 0.9e-3, 0.9e-3    # beam 1/e^2 radii [m]
    wa_x, wa_y, wa_z = 35e-6, 35e-6, 100e-6       # cloud 1-sigma radii [m]
    tx, ty, tz       = 5e-6, 5e-6, 5e-6            # cloud temperatures [K]

    V_pd = 70
    P_689 = (0.5 + 0.229 * V_pd) * 1e-3
    P_688 = 0.0     # off for single-photon dynamics
    P_679 = 0.0

    # Beam directions
    theta_0, theta_0z = np.radians(59.4384), 0.0
    theta_1, theta_1z = np.radians(-59.64),  0.0
    theta_2, theta_2z = 0.0, 0.0

    k_vec_0 = (2*PI / lambda_689) * get_k_hat(theta_0, theta_0z)
    k_vec_1 = (2*PI / lambda_688) * get_k_hat(theta_1, theta_1z)
    k_vec_2 = (2*PI / lambda_679) * get_k_hat(theta_2, theta_2z)

    k_vecs     = (k_vec_0, k_vec_1, k_vec_2)
    powers     = (P_689, P_688, P_679)
    beam_radii = (w0, w1, w2)

    # Quantization axis
    quant_axis = np.array([-1.0, 0.0, 0.0])    # -x_hat

    # Polarization unit vectors for each beam in the lab frame
    eps_0 = np.array([0.0, 0.0, 1.0])           # z_hat  (689 nm beam)
    eps_1 = np.array([0.0, 0.0, 1.0])           # placeholder (688 nm beam)
    eps_2 = np.array([-1.0, 0.0, 0.0])           # placeholder (679 nm beam)
    pol_vecs = (eps_0, eps_1, eps_2)

    # Target Zeeman sublevel for each beam
    mJ_targets = (+1, 0, 0) 

    # --- Zeeman detuning ---
    B_field_G = 20   # bias field magnitude [G]
    B_field_T = B_field_G * 1e-4
    delta_zeeman_689 = get_zeeman_detuning(G_J_3P1, mJ_targets[0], B_field_T)
    print(f"Zeeman shift (mJ={mJ_targets[0]:+d}): "
          f"2pi x {delta_zeeman_689/(2*PI)*1e-3:.1f} kHz at B = {B_field_G:.1f} G")

    detunings = [0.0, 0.0, 0.0]   # laser is on resonance with each beam's target sublevel

    N_atoms = 50

    # Sample ensemble; atleast_2d ensures shape (N, 3) for vectorized functions
    pos, vel = sample_atomic_ensemble([wa_x, wa_y, wa_z], [tx, ty, tz], n_samples=N_atoms)
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)

    T_MAX = 0.35e-6

    # Simulate and plot 1-photon Rabi flopping
    tlist, avg_pop = simulate_one_photon_rabi_dynamics(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9
    )
    _, avg_pop_ideal = simulate_one_photon_rabi_dynamics(
        np.zeros((1, 3)), np.zeros((1, 3)), beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9
    )

    # Theory curve: coupling factor applied, damped envelope included
    C0           = get_coupling_factor(eps_0, quant_axis, mJ_targets[0])
    Omega_theory = C0 * gamma_689 * np.sqrt(P_689 * 100 / (PI*w0**2) / Is_689)
    Omega_eff    = np.sqrt(Omega_theory**2 + detunings[0]**2)
    
    decay_env    = np.exp(-3/4 * tlist * gamma_689)
    pop_theory   = (Omega_theory**2 / Omega_eff**2) * (0.5 - 0.5 * np.cos(Omega_eff * tlist) * decay_env)

    # Readout model
    t_push          = 0.8e-6
    avg_pop_meas    = apply_readout(avg_pop,    t_push)
    pop_theory_meas = apply_readout(pop_theory, t_push)
    avg_pop_meas_ideal = apply_readout(avg_pop_ideal, t_push)

    I_peak          = 2 * P_689 * 100 / (PI * w0**2)           # peak intensity [uW/cm^2]
    Omega_bare      = gamma_689 * np.sqrt(I_peak / (2 * Is_689)) # Rabi freq before coupling factor
    T_rabi          = 2*PI / Omega_eff                           # Rabi period [s]
    peak_excitation = Omega_theory**2 / Omega_eff**2             # on-resonance max population (no decay)
    readout_fidelity = np.exp(-gamma_689 * t_push)

    print("=" * 55)
    print("  BEAM & POLARIZATION")
    print(f"    Peak intensity (beam center):  {I_peak:.1f} uW/cm^2")
    print(f"    Saturation intensity (Is_689): {Is_689:.2f} uW/cm^2")
    print(f"    Saturation parameter  s=I/Is:  {I_peak/Is_689:.1f}")
    print(f"    Coupling factor C = |eps_q|:   {C0:.4f}  "
          f"(mJ={mJ_targets[0]:+d})")
    print()
    print("  RABI DYNAMICS")
    print(f"    Bare Rabi freq  (C=1):         2pi x {Omega_bare/(2*PI)*1e-6:.3f} MHz")
    print(f"    Rabi freq       (with C):      2pi x {Omega_theory/(2*PI)*1e-6:.3f} MHz")
    print(f"    Zeeman shift    (mJ={mJ_targets[0]:+d}):      2pi x {delta_zeeman_689/(2*PI)*1e-3:.1f} kHz  at B={B_field_G:.1f} G")
    print(f"    Laser freq offset from bare:   2pi x {(detunings[0]+delta_zeeman_689)/(2*PI)*1e-3:.1f} kHz  [lab tuning ref, not in sim]")
    print(f"    Sim detuning from mJ={mJ_targets[0]:+d} level: 2pi x {detunings[0]/(2*PI)*1e-3:.1f} kHz  [detunings[0] -> enters Hamiltonian]")
    print(f"    Effective Rabi freq:           2pi x {Omega_eff/(2*PI)*1e-6:.3f} MHz")
    print(f"    Rabi period:                   {T_rabi*1e6:.3f} us")
    print(f"    pi-pulse time:                 {T_rabi/2*1e6:.3f} us")
    print(f"    Peak excitation (no decay):    {peak_excitation:.4f}")
    print()
    print("  READOUT")
    print(f"    Push duration:                 {t_push*1e6:.2f} us")
    print(f"    Readout fidelity:              {readout_fidelity:.4f}  ({readout_fidelity:.1%})")
    print(f"    Max observable population:     {peak_excitation * readout_fidelity:.4f}")
    print("=" * 55)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tlist * 1e6, avg_pop_meas,    color='C0', label='Simulation')
    ax.plot(tlist * 1e6, avg_pop_meas_ideal,    color='C1', label='Ideal')
    # ax.plot(tlist * 1e6, pop_theory_meas, color='C2', linestyle='--', label='Theory')
    ax.scatter(t_raw_us, pop_avg_raw, color='C3', label='Raw Data')

    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Excited state population")
    ax.set_title("689 nm Single-Photon Rabi Flopping")
    # ax.set_ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.show()

