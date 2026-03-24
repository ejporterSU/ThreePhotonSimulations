import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from tqdm import tqdm

from simulation_functions import *

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