import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from tqdm import tqdm
import h5py

from simulation_functions import *

def read_RID(rid):    
    with h5py.File(f'Data/0000{rid}-ClockExcitation_exp.h5', 'r') as f:
        x = f['datasets']['current_scan.plots.x'][:]
        y = f['datasets']['current_scan.plots.y'][:]
    return x, y

#%%

if __name__ == "__main__":
    B_field_G    = 20   # bias field magnitude [G]
    detunings    = [2*PI * 5e6, 0.0, 0.0]   # laser is on resonance with each beam's target sublevel
    T_MAX        = 1e-6
    t_push       = 0.8e-6
    N_atoms      = 50
    sigma_aom = 90e-9





    w0, w1, w2       = 0.5e-3, 0.9e-3, 0.9e-3    # beam 1/e^2 radii [m]
    wa_x, wa_y, wa_z = 30e-6, 0e-6, 0e-6       # cloud 1-sigma radii [m]
    tx, ty, tz       = 0e-6, 0e-6, 0e-6            # cloud temperatures [K]

    P_689 = 10e-3 # peak power
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
    B_field_T = B_field_G * 1e-4
    delta_zeeman_689 = get_zeeman_detuning(G_J_3P1, mJ_targets[0], B_field_T)
    print(f"Zeeman shift (mJ={mJ_targets[0]:+d}): "
          f"2pi x {delta_zeeman_689/(2*PI)*1e-3:.1f} kHz at B = {B_field_G:.1f} G")

    


    # Sample ensemble; atleast_2d ensures shape (N, 3) for vectorized functions
    pos, vel = sample_atomic_ensemble([wa_x, wa_y, wa_z], [tx, ty, tz], n_samples=N_atoms)
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)



    # Simulate and plot 1-photon Rabi flopping
    tlist, avg_pop = simulate_one_photon_rabi_dynamics(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9
    )

    # AOM pulse: 50% rise at t=0 (matching experiment), ramp starts at ~-sigma
    aom_params = {"t0": 0.0, "sigma": sigma_aom, "t_pulse": T_MAX}
    _, avg_pop_aom = simulate_one_photon_rabi_dynamics(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9,
        aom_params=aom_params
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
    
    avg_pop_meas     = apply_readout(avg_pop,     t_push)
    avg_pop_aom_meas = apply_readout(avg_pop_aom, t_push)
    # pop_theory_meas  = apply_readout(pop_theory,  t_push)
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

    # Pulse-shape panels: show full envelope for t_pulse = 1x, 2x, 5x sigma

    t_us        = tlist * 1e6
    pulse_cases = [(1, '1σ'), (2, '2σ'), (5, '5σ ')]

    # Layout: 3 pulse-shape panels on top, main Rabi plot spanning full width below
    fig = plt.figure(figsize=(10, 6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 2.5], hspace=0.5, wspace=0.35)

    ax_b    = fig.add_subplot(gs[0, 0])
    ax_m    = fig.add_subplot(gs[0, 1])
    ax_e    = fig.add_subplot(gs[0, 2])
    ax_main = fig.add_subplot(gs[1, :])

    for ax_p, (n, title) in zip([ax_b, ax_m, ax_e], pulse_cases):
        t_pulse_i  = n * sigma_aom
        t_end      = 6 * sigma_aom + t_pulse_i          # 3σ rise + plateau + 3σ fall
        t_panel    = np.linspace(0, t_end, 500)
        coeff_p, _ = aom_rabi_envelope(t0=0.0, sigma=sigma_aom,
                                       t_pulse=t_pulse_i, Omega_peak=1.0)
        env_p      = np.array([coeff_p(t) for t in t_panel])
        ax_p.plot(t_panel * 1e9, env_p, color='C1')
        ax_p.fill_between(t_panel * 1e9, env_p, alpha=0.2, color='C1')
        ax_p.set_title(f't_pulse = {title}', fontsize=9)
        ax_p.set_xlabel('Time [ns]', fontsize=8)
        ax_p.set_ylabel(r'$\Omega(t)/\Omega_0$', fontsize=8)
        ax_p.set_ylim(0, 1.15)
        ax_p.tick_params(labelsize=7)

    # Main Rabi flop plot
    ax_main.plot(t_us, avg_pop_meas,       color='C0', label='Square pulse')
    ax_main.plot(t_us, avg_pop_aom_meas,   color='C1', linestyle='--',
                 label=rf'AOM pulse ($\sigma$ = {sigma_aom*1e9} ns)')
    # ax_main.plot(t_us, avg_pop_meas_ideal, color='C2', label='Ideal')
    # ax_main.plot(t_us, pop_theory_meas, color='C3', linestyle='--', label='Theory')
    # ax_main.scatter(t_raw_us, pop_avg_raw, color='C4', label='Raw Data')

    ax_main.set_xlabel("Time [us]")
    ax_main.set_ylabel("Excited state population")
    ax_main.set_title("689 nm Single-Photon Rabi Flopping")
    # ax_main.set_ylim(0, 1)
    ax_main.legend()

    plt.tight_layout()
    plt.show()


