#%%
"""
Sr-88 three-photon excitation: 1S0 -> 3P1(mJ=+1) -> 3S1(mJ=0) -> 3P0
Basic QuTiP Lindblad master equation structure.

State ordering:
|0> = 1S0            (ground)
|1> = 3P1, mJ=-1     (intermediate 1, driven by 689 nm sigma-)
|2> = 3P1, mJ=+1     (intermediate 1, driven by 688 nm sigma+)
|3> = 3S1, mJ=0      (intermediate 2, driven by 688 nm pi)
|4> = 3P2, all mJ    (dark decay channel 3P2)
|5> = 3P0            (clock state)

"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from scipy.special import erf
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────── #
#region
HBAR = const.hbar
H    = const.h
C    = const.c
PI   = np.pi
AMU  = 1.66e-27   # atomic mass unit [kg]
kb   = const.k    # Boltzmann constant [J/K]
MU_B = const.physical_constants['Bohr magneton'][0]  # Bohr magneton [J/T]
EPS0 = const.epsilon_0
#endregion

# -- HELPERS ------------------
#region
def sample_atomic_ensemble(radii, temperatures, mass=88*AMU, n_samples=1):
    """Sample positions and velocities from a thermal Gaussian cloud.

    Args:
        radii:        1-sigma cloud radius [m]. Scalar for isotropic, or [x,y,z] array.
        temperatures: Temperature [K]. Scalar for isotropic, or [x,y,z] array.
        mass:         Atomic mass [kg]. Defaults to Sr-88.
        n_samples:    Number of atoms to draw.

    Returns:
        positions  (n_samples, 3) array [m]
        velocities (n_samples, 3) array [m/s]
    """
    sigma_r = np.array(radii)
    sigma_v = np.sqrt(kb * np.array(temperatures) / mass)

    positions  = np.random.normal(loc=0.0, scale=sigma_r, size=(n_samples, 3))
    velocities = np.random.normal(loc=0.0, scale=sigma_v, size=(n_samples, 3))

    if n_samples == 1:
        return np.atleast_2d(positions[0]), np.atleast_2d(velocities[0])
    return np.atleast_2d(positions), np.atleast_2d(velocities)

def get_k_hat(theta, theta_z):
    """Return a unit wavevector in the azimuth-elevation parameterisation.

    Args:
        theta:   Azimuthal angle in the x-y plane [rad].
        theta_z: Elevation angle above the x-y plane [rad].

    Returns:
        (3,) unit vector [cos(θ_z)cos(θ), cos(θ_z)sin(θ), sin(θ_z)].
    """
    return np.array([np.cos(theta_z)*np.cos(theta),
                     np.cos(theta_z)*np.sin(theta),
                     np.sin(theta_z)])


def get_effective_r_perp(pos, k_vec):
    """Compute each atom's perpendicular distance from a beam axis.

    Args:
        pos:   (N, 3) atom positions [m].
        k_vec: (3,) beam wavevector (any magnitude; only direction is used).

    Returns:
        (N,) perpendicular distances from the beam axis [m].
    """
    k_hat      = k_vec / np.linalg.norm(k_vec)
    proj_mag   = np.sum(pos * k_hat, axis=1)   # scalar projection onto beam axis, (N,)
    r_parallel = np.outer(proj_mag, k_hat)      # parallel component, (N, 3)
    r_perp_vec = pos - r_parallel               # perpendicular component, (N, 3)
    return np.linalg.norm(r_perp_vec, axis=1)

def get_calculated_parameters(position, velocity, k_vecs, omegas, beam_radii):
    """Compute per-atom Doppler shifts and beam-attenuated Rabi frequencies.

    Doppler shift for beam i: δ_i = -k⃗_i · v⃗  (add to bare detuning in H_diag).
    Rabi frequency for beam i: Ω_i(r) = Ω_i,0 * exp(-r_perp² / w_i²), where
    r_perp is the atom's distance from the beam axis and w_i is the 1/e² radius.

    Args:
        position:   (N, 3) atom positions [m].
        velocity:   (N, 3) atom velocities [m/s].
        k_vecs:     Tuple of three (3,) full wavevectors [rad/m] (magnitude = 2π/λ).
        omegas:     Tuple of three peak Rabi frequencies (Ω_689, Ω_688, Ω_679) [rad/s].
        beam_radii: Array of three 1/e² beam radii [m].

    Returns:
        Dict with keys 'beam_0', 'beam_1', 'beam_2', each containing:
            'dshift': (N,) Doppler shift [rad/s]
            'Omega':  (N,) attenuated Rabi frequency [rad/s]
    """
    k_vec_0, k_vec_1, k_vec_2 = k_vecs
    rabi_0, rabi_1, rabi_2 = omegas
    w0, w1, w2 = beam_radii

    doppler_0 = -np.sum(k_vec_0 * velocity, axis=1)
    doppler_1 = -np.sum(k_vec_1 * velocity, axis=1)
    doppler_2 = -np.sum(k_vec_2 * velocity, axis=1)

    r_perp_0 = get_effective_r_perp(position, k_vec_0)
    r_perp_1 = get_effective_r_perp(position, k_vec_1)
    r_perp_2 = get_effective_r_perp(position, k_vec_2)

    rabi_0 *= np.exp(-r_perp_0**2 / w0**2)
    rabi_1 *= np.exp(-r_perp_1**2 / w1**2)
    rabi_2 *= np.exp(-r_perp_2**2 / w2**2)

    return {
        "beam_0": {"dshift": doppler_0, "Omega": rabi_0},
        "beam_1": {"dshift": doppler_1, "Omega": rabi_1},
        "beam_2": {"dshift": doppler_2, "Omega": rabi_2},
    }


#endregion



# --- Atomic Parameters ---------------------------------------
#region
gamma_689, lambda_689 = 2*PI * 7.48e3,  689.4489e-9
gamma_688, lambda_688 = 2*PI * 3.90e6,  688.020770e-9
gamma_679, lambda_679 = 2*PI * 1.26e6,  679.288943e-9
gamma_707, lambda_707 = 2*PI * 6.225e6, 707.197215e-9

# ── Basis states ──────────────────────────────────────────────────────────── #
#region

N  = 6
g = qt.basis(N, 0)   # |1S0>
e1 = qt.basis(N, 1)   # |3P1, mJ=-1>
e3 = qt.basis(N, 2)   # |3P1, mJ=+1>
v2 = qt.basis(N, 3)   # |3S1, mJ=0>
ds = qt.basis(N, 4)   # |3P2, mJ=all> dark state decay channel
r = qt.basis(N, 5)   # |3P0>

states = [g, e1, e3, v2, ds, r]
projs = [state * state.dag() for state in states]
#endregion

#---- experimental parameters ------------------------------------------
#region

theta_0,  theta_0z = np.radians(59.4384), 0.0   # 689 nm
theta_1,  theta_1z = np.radians(-59.64),  0.0   # 688 nm
theta_2,  theta_2z = 0.0,                 0.0   # 679 nm (along x)

# full wavevectors [rad/m]: magnitude 2π/λ so that k⃗·v⃗ gives the Doppler shift in rad/s
k_vec_0 = (2*PI / lambda_689) * get_k_hat(theta_0, theta_0z)
k_vec_1 = (2*PI / lambda_688) * get_k_hat(theta_1, theta_1z)
k_vec_2 = (2*PI / lambda_679) * get_k_hat(theta_2, theta_2z)
k_vecs  = (k_vec_0, k_vec_1, k_vec_2)

w0_689       = 0.54e-3       # 689 nm 1/e^2 radius [m]
w0_688       = 0.90e-3       # 688 nm 1/e^2 radius [m]
w0_679       = 0.90e-3       # 679 nm 1/e^2 radius [m]
beam_radii = np.array([w0_689, w0_688, w0_679])

# Rabi frequencies [rad/s] 
Omega_689 =  2*PI * 3.75e6    # 689 nm:  1S0  <-> 3P1(mJ=+1)
Omega_688 =  2*PI * 21.6e6     # 688 nm:  3P1  <-> 3S1(mJ=0)
Omega_679 =  2*PI * 21.4e6     # 679 nm:  3S1  <-> 3P0
dwB_3p1 =    2*PI * 40e6

# Single-photon detunings from each resonance [rad/s] 
delta_AC = 2*PI * -0.395e6  # determined from FREQ mode scans
dwB_3p1  = 2*PI * 40e6
delta_1  = 2*PI * 45e6
delta_2  = 2*PI * 355e6

# cumulative detunings
Delta_1 = delta_1   # 689 nm detuning from 1S0 -> 3P1
Delta_2 = delta_1 + delta_2   # 688 nm detuning from 3P1 -> 3S1
Delta_3 = delta_AC # nominally on resonance, so gets set to the ac stark


# ── sim params ────────────────────────────────────────────────────────── #
N_atoms  = 500  # set to 1 if you dont want to use a thermal, finite size cloud
T_atom   = [8e-6,  8e-6,  5e-6]   # atom temperature [K] per axis (x, y, z)
sigma_r  = [40e-6, 40e-6, 120e-6]  # cloud 1-sigma radius [m] per axis (x, y, z)

MODE='TIME'  # set to FREQ to find AC stark
USE_RAMP  = True  # False for square wave, True for AOM shapes
PLOT_ENVELOPE = False   # set True to preview the drive envelope before running
T_MAX  = 10e-6   # total time [s]
dt = 20e-9  # simulation resolution
N_t    = int(T_MAX/dt) + 1
tlist  = np.linspace(0, T_MAX, N_t)
ac_starks = 2*PI*1e6*np.linspace(-0.5, -0.3, 20)  # used for finding the AC stark, zoom in as needed

N_t_ramp = 50  # number of ramped points to simulate (different from simulation resolution)

def drive_envelope(t, args):
    if USE_RAMP:
        T_pulse = args.get('T_total', T_MAX)  # pulse window end [s]

        # emprically determined AOM shaping, prob dont touch
        t_rise = 100e-9              # AOM turn-on delay: 50% rise point
        t_fall = T_pulse + 45e-9    # AOM turn-off offset: 50% fall point
        tau_ramp  = 80e-9   # erf width [s]; rise/fall spans for AOM pulse shaping
        ramp_up   = 0.5 * (1.0 + erf((t - t_rise) / tau_ramp))
        ramp_down = 0.5 * (1.0 - erf((t - t_fall) / tau_ramp))


        return ramp_down * ramp_up
    return 1.0  # for square wave


if PLOT_ENVELOPE and USE_RAMP: # just for visualizing rabi envelopes
    preview_t_ons = [0, T_MAX / 4, T_MAX / 2, T_MAX * 3 / 4, T_MAX]
    fig_env, ax_env = plt.subplots(figsize=(7, 3))
    for t_on in preview_t_ons:
        T_sim   = t_on + 150e-9   # extra time past t_fall (= t_on+45ns) for AOM ramp to reach ~0
        t_plot = np.linspace(0, T_sim, 2000)
        env = np.array([drive_envelope(t, {'T_total': t_on}) for t in t_plot])
        ax_env.plot(t_plot * 1e6, env, label=f"t_on = {t_on*1e6:.2f} µs")
    ax_env.set_xlabel('Time [µs]')
    ax_env.set_ylabel('Envelope')
    ax_env.set_title('Drive envelope preview')
    ax_env.set_ylim(-0.05, 1.15)
    ax_env.legend()
    plt.tight_layout()
    plt.show()


# initial state
rho0 = g*g.dag()
# ── Collapse operators (Lindblad spontaneous emission) ────────────────────── #
c_3P1_to_1S0 = [np.sqrt(gamma_689) * (g*e1.dag()), np.sqrt(gamma_689) * (g*e3.dag()) ]
c_3S1_to_3P1 = [np.sqrt(gamma_688/2) * (e1*v2.dag()), np.sqrt(gamma_688/2) * (e3*v2.dag())]  # /2 per channel: two equal branches (→e1, →e3) sum to gamma_688
c_3S1_to_3P0 = [np.sqrt(gamma_679) * (r*v2.dag())]
c_3S1_to_3P2 = [np.sqrt(gamma_707) * (ds*v2.dag())]
c_ops = c_3P1_to_1S0 + c_3S1_to_3P1 + c_3S1_to_3P0 + c_3S1_to_3P2

if MODE=='TIME': # for performing rabi scans
    if N_atoms == 1:  # 0 temp, 0 cloud size assuming
        # rotating frame: diagonal entry is -Δ_i for each state, where Δ_i is the
        # cumulative laser detuning from that state's resonance (+ Zeeman shift where applicable)
        H_diag = (
            - (Delta_1 + dwB_3p1           )  * projs[1]
            - (Delta_1 - dwB_3p1           )  * projs[2]
            - (Delta_2                     )  * projs[3]
            - (0                           )  * projs[4] # no coupled drive, just let it sit in its own frame
            - (Delta_3                     )  * projs[5]

        )

        # Coherent couplings (Omega/2 for each laser field)
        H_689 = Omega_689/2 * (e1*g.dag() + g*e1.dag() +
                            e3*g.dag() + g*e3.dag() )

        H_688 = Omega_688/2 * (e1*v2.dag() + v2*e1.dag() +
                            e3*v2.dag() + v2*e3.dag() )

        H_679 = Omega_679/2 * (v2*r.dag() + r*v2.dag())
        H_coupling = H_688 + H_679 + H_689

    
        

        if USE_RAMP: # AOM pulse shaping

            t_on_list = np.linspace(0, T_MAX, N_t_ramp)  # list of points to simulate ramp for
            H = [H_diag, [H_coupling, drive_envelope]]  # time dependent hamiltonian
            final_pops = [[] for _ in range(len(states))]  # store populations at end of ramp (i.e. final measurement)

            for t_on in tqdm(t_on_list, desc="Scanning on-time..."):
                T_sim   = t_on + 150e-9   # sim end: give extra time for ramp up and down
                n_pts = max(int(T_sim / dt) + 1, 20)  # based on sim res
                tlist_i = np.linspace(0, T_sim, n_pts)
                result = qt.mesolve(H, rho0, tlist_i, c_ops, e_ops=projs,
                                    args={'T_total': t_on})  # simulate ensemble
                for i, pop in enumerate(result.expect):
                    final_pops[i].append(pop[-1])
            t_plot = t_on_list
            

        else:
            # just square shape pulse
            H = H_diag + H_coupling
            result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs)
            final_pops   = result.expect
            t_plot = tlist


        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['g', 'em1', 'e1', 'v0', 'ds', 'r']
        colors = [f'C{i}' for i in range(6)]
        for pop, label, color in zip(final_pops, labels, colors):
            ax.plot(t_plot * 1e6, pop, color=color, label=f"{label} {max(np.abs(pop)):.3f}")
        ax.set_xlabel('Time [µs]')
        ax.set_ylabel('Population')
        ax.set_title('Sr-88 Three Photon Simulation')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:  # for ensemble averaging 
        if USE_RAMP:
            t_on_list = np.linspace(0, T_MAX, N_t_ramp)
            ensemble_pops = [np.zeros(N_t_ramp) for _ in range(len(states))]

            for j, t_on in enumerate(tqdm(t_on_list, desc="Scanning on-time...")):
                positions, velocities = sample_atomic_ensemble(sigma_r, T_atom, n_samples=N_atoms)
                params = get_calculated_parameters(positions, velocities, k_vecs,
                                                   (Omega_689, Omega_688, Omega_679), beam_radii)
                T_sim = t_on + 150e-9
                n_pts = max(int(T_sim / dt) + 1, 20)
                tlist_i = np.linspace(0, T_sim, n_pts)

                for atom_idx in range(N_atoms):
                    ds_0 = params["beam_0"]["dshift"][atom_idx]
                    ds_1 = params["beam_1"]["dshift"][atom_idx]
                    ds_2 = params["beam_2"]["dshift"][atom_idx]
                    O_0  = params["beam_0"]["Omega"][atom_idx]
                    O_1  = params["beam_1"]["Omega"][atom_idx]
                    O_2  = params["beam_2"]["Omega"][atom_idx]

                    H_diag_i = (
                        - (Delta_1 + dwB_3p1 + ds_0) * projs[1]
                        - (Delta_1 - dwB_3p1 + ds_0) * projs[2]
                        - (Delta_2           + ds_1 ) * projs[3]
                        - (0                        ) * projs[4]
                        - (Delta_3           + ds_2 ) * projs[5]
                    )
                    H_coupling_i = (O_0/2 * (e1*g.dag() + g*e1.dag() + e3*g.dag() + g*e3.dag()) +
                                     O_1/2 * (e1*v2.dag() + v2*e1.dag() + e3*v2.dag() + v2*e3.dag()) +
                                     O_2/2 * (v2*r.dag() + r*v2.dag()))
                    H_i = [H_diag_i, [H_coupling_i, drive_envelope]]

                    result = qt.mesolve(H_i, rho0, tlist_i, c_ops, e_ops=projs,
                                        args={'T_total': t_on})
                    for si, pop in enumerate(result.expect):
                        ensemble_pops[si][j] += pop[-1]

            for si in range(len(states)):
                ensemble_pops[si] /= N_atoms

            fig, ax = plt.subplots(figsize=(8, 4))
            labels = ['g', 'em1', 'e1', 'v0', 'ds', 'r']
            colors = [f'C{i}' for i in range(6)]
            for pop, label, color in zip(ensemble_pops, labels, colors):
                ax.plot(t_on_list * 1e6, pop, color=color, label=f"{label} {max(np.abs(pop)):.3f}")
                ax.scatter(t_on_list * 1e6, pop, color=color)
            ax.set_xlabel('On Time [µs]')
            ax.set_ylabel('Population')
            ax.set_title(f'Sr-88 Three Photon Simulation (N={N_atoms} atoms)')
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            plt.tight_layout()
            plt.show()

        else:  # square wave
            positions, velocities = sample_atomic_ensemble(sigma_r, T_atom, n_samples=N_atoms)
            params = get_calculated_parameters(positions, velocities, k_vecs,
                                               (Omega_689, Omega_688, Omega_679), beam_radii)
            ensemble_pops = [np.zeros(N_t) for _ in range(len(states))]

            for atom_idx in tqdm(range(N_atoms), desc="Ensemble atoms..."):
                ds_0 = params["beam_0"]["dshift"][atom_idx]
                ds_1 = params["beam_1"]["dshift"][atom_idx]
                ds_2 = params["beam_2"]["dshift"][atom_idx]
                O_0  = params["beam_0"]["Omega"][atom_idx]
                O_1  = params["beam_1"]["Omega"][atom_idx]
                O_2  = params["beam_2"]["Omega"][atom_idx]

                H_diag_i = (
                    - (Delta_1 + dwB_3p1 + ds_0) * projs[1]
                    - (Delta_1 - dwB_3p1 + ds_0) * projs[2]
                    - (Delta_2           + ds_1 ) * projs[3]
                    - (0                        ) * projs[4]
                    - (Delta_3           + ds_2 ) * projs[5]
                )
                H_coupling_i = (O_0/2 * (e1*g.dag() + g*e1.dag() + e3*g.dag() + g*e3.dag()) +
                                 O_1/2 * (e1*v2.dag() + v2*e1.dag() + e3*v2.dag() + v2*e3.dag()) +
                                 O_2/2 * (v2*r.dag() + r*v2.dag()))
                H_i = H_diag_i + H_coupling_i
                result = qt.mesolve(H_i, rho0, tlist, c_ops, e_ops=projs)
                for si, pop in enumerate(result.expect):
                    ensemble_pops[si] += np.array(pop)

            for si in range(len(states)):
                ensemble_pops[si] /= N_atoms

            fig, ax = plt.subplots(figsize=(8, 4))
            labels = ['g', 'em1', 'e1', 'v0', 'ds', 'r']
            colors = [f'C{i}' for i in range(6)]
            for pop, label, color in zip(ensemble_pops, labels, colors):
                ax.plot(tlist * 1e6, pop, color=color, label=f"{label} {max(np.abs(pop)):.3f}")
            ax.set_xlabel('Time [µs]')
            ax.set_ylabel('Population')
            ax.set_title(f'Sr-88 Three Photon Simulation (N={N_atoms} atoms)')
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            plt.tight_layout()
            plt.show()

# for finding AC stark
# always does with ideal condition (square wave, T=0, sigma_c=0)
if MODE=='FREQ':
    
    max_clock = []  # stores peak clock excitation

    # Coherent couplings (Omega/2 for each laser field)
    H_689 = Omega_689/2 * (e1*g.dag() + g*e1.dag() + 
                        e3*g.dag() + g*e3.dag() )

    H_688 = Omega_688/2 * (e1*v2.dag() + v2*e1.dag() + 
                        e3*v2.dag() + v2*e3.dag() ) 

    H_679 = Omega_679/2 * (v2*r.dag() + r*v2.dag())
    H_coupling = H_688 + H_679 + H_689


    for dac in tqdm(ac_starks, desc="Running Stark Scan..."):
        H_diag = (
        - (Delta_1 + dwB_3p1           )  * projs[1]
        - (Delta_1 - dwB_3p1           )  * projs[2]
        - (Delta_2                     )  * projs[3]
        - (0                           )  * projs[4] # no coupled drive, just let it sit in its own frame
        - (dac                         )  * projs[5]
 
        )
        
        H = H_diag + H_coupling
        # ── Time evolution ────────────────────────────────────────────────────────── #
        result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs[-1])
        pops   = result.expect 
        max_clock.append(max(pops[0]))

    plt.plot(ac_starks / (2*PI*1e6), max_clock)  # shows peak clock excitation as a function of delta_3

    imax = np.argmax(max_clock)  # find peak and visualize
    plt.axvline(ac_starks[imax]/(2*PI*1e6), label=f"Peak: {ac_starks[imax]/(2*PI*1e6):0.3f} MHz")

    plt.legend()
    plt.xlabel("AC Stark [MHz]")
    plt.ylabel("Max Population")



#%%

# for saving to a text file, will need to change what gets saved depending on scenario probably
# pops = np.array(ensemble_pops)
# state_names = ['1S0', '3P1(mJ=-1)', '3P1(mJ+1)', '3S1(mJ=0)', '3P2(all)', '3P0']
# header = 'T [us]  ' + ', '.join(state_names)
# data   = np.column_stack([t_on_list * 1e6, pops.T])
# np.savetxt('sim_pops.txt', data, header=header, fmt='%.6e', delimiter=',')
