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
    return x, y*1e-6


def get_power_mw(pd_mv):
    # returns 689 power as a function of photodiode peak power
    return 0.229*pd_mv + 0.5

#%%
if __name__ == "__main__":

    t_raw, pop_raw = read_RID(75207) ## read in RID


    B_field_G    = 20   # bias field magnitude [G]
    detunings    = 2*PI*1e6 * np.array([5.0, 0.0, 0.0])   # laser is on resonance with each beam's target sublevel
    T_MAX        = 0.3e-6
    t_push       = 0.8e-6
    N_atoms      = 100
    sigma_aom = 3000e-9
    pd_mv = 75

    aom_params = {"t0": 0.0, "sigma": sigma_aom, "t_pulse": T_MAX}





    w0, w1, w2       = 0.58e-3, 0.9e-3, 0.9e-3    # beam 1/e^2 radii [m]
    wa_x, wa_y, wa_z = 43e-6, 43e-6, 127e-6       # cloud 1-sigma radii [m]
    tx, ty, tz       = 3e-6, 3e-6, 6.5e-6            # cloud temperatures [K]
    P_fudge = 1
    P_689 = P_fudge*get_power_mw(pd_mv)*1e-3 # peak power
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



    tlist, avg_pop_aom = simulate_one_photon_rabi_dynamics(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9,
        aom_params=aom_params
    )
    tlist, avg_pop = simulate_one_photon_rabi_dynamics(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=5e-9
    )

    # Theory curve: coupling factor applied, damped envelope included
    C0           = get_coupling_factor(eps_0, quant_axis, mJ_targets[0])
    Omega_theory = C0 * gamma_689 * np.sqrt(P_689 * 100 / (PI*w0**2) / Is_689)
    Omega_eff    = np.sqrt(Omega_theory**2 + detunings[0]**2)

    # Readout model
    avg_pop_meas     = apply_readout(avg_pop,     t_push)
    avg_pop_meas_aom     = apply_readout(avg_pop_aom,     t_push)


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

    # Simulation vs experimental data
    # t_raw assumed in seconds (ARTIQ convention); change to t_raw if already in us
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tlist * 1e6, avg_pop_meas_aom, color='C0', label='Simulation (AOM)')
    ax.plot(tlist * 1e6, avg_pop_meas, color='C1', label='Simulation (no AOM)')
    ax.scatter(t_raw * 1e6, pop_raw, color='C3', s=25, zorder=5,
               label='Exp data (RID 75202)')
    print(t_raw)
    ax.set_xlabel('Pulse duration [us]')
    ax.set_ylabel('Excited state population')
    ax.set_title(f'689 nm Rabi  —  P = {P_689*1e3:.2f} mW, '
                 f'pd = {pd_mv} mV')
    ax.legend()

    plt.tight_layout()
    plt.show()


