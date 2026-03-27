#%%

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from tqdm import tqdm
import h5py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from simulation_functions import *


def read_RID(rid, exp_name="ClockExcitation"):    
    with h5py.File(f'Data/0000{rid}-{exp_name}_exp.h5', 'r') as f:
        x = f['datasets']['current_scan.plots.x'][:]
        y = f['datasets']['current_scan.plots.y'][:]
    return x, y*1e-6


def get_power_mw(pd_mv):
    # returns 689 power as a function of photodiode peak power
    return 0.229*pd_mv - 0.5


if __name__ == "__main__":
    # params for validates
    rids = [75202, 75205, 75207, 75208, 75210]
    d689s = 2*PI*1e6 * np.array([0, 1, 5, 5, 3])
    pd_mvs = np.array([85, 85, 75, 75, 85])

    raw_data = [(read_RID(rids[i])) for i in range(5)]
    t_raws, pop_raws = zip(*raw_data) # unpack/unzip into seperate lists


    B_field_G    = 20   # bias field magnitude [G]
    detunings    = 2*PI*1e6 * np.array([0.0, 0.0, 0.0])   # laser is on resonance with each beam's target sublevel
    T_MAX        = 1e-6
    t_push       = 0.8e-6
    N_atoms      = 75
    sigma_aom = 90e-9
    pd_mv = 85
    P_fudge = 0.9

    aom_params = {"t0": 0.0, "sigma": sigma_aom}

    w0, w1, w2       = 0.54e-3, 0.9e-3, 0.9e-3    # beam 1/e^2 radii [m]
    wa_x, wa_y, wa_z = 43e-6, 43e-6, 127e-6       # cloud 1-sigma radii [m]
    tx, ty, tz       = 2.5e-6, 2.5e-6, 6.5e-6            # cloud temperatures [K]
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

    k_vecs     = [k_vec_0, k_vec_1, k_vec_2]
    powers     = [P_689, P_688, P_679]
    cloud_radii = [wa_x, wa_y, wa_z]
    beam_radii = [w0, w1, w2]
    temp_vec = [tx, ty, tz]

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
    

    


    # Sample ensemble; atleast_2d ensures shape (N, 3) for vectorized functions
    pos, vel = sample_atomic_ensemble(cloud_radii, temp_vec, n_samples=N_atoms)
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)


    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(5):
        # update params
        detunings[0] = d689s[i]
        powers[0] = P_fudge*get_power_mw(pd_mvs[i])*1e-3


        output_times, output_pops = simulate_one_photon_rabi_dynamics(
            pos, vel, beam_radii, powers, detunings, k_vecs,
            pol_vecs, quant_axis, mJ_targets, t_max=T_MAX, dt=20e-9,
            envelope='ERF', envelope_params=aom_params
        )

        output_pops = apply_readout(output_pops,     t_push) # push pulse model

        ax.plot(output_times * 1e6, output_pops, color=f'C{i+1}') 


        if  i == 3:
            x, y = t_raws[i] * 1e6, pop_raws[i]
            x = x[np.where(y<0.2)]
            y = y[np.where(y<0.2)]

            ax.scatter( x, y, color=f'C{i+1}', s=25,
                label=f'RID: {rids[i]}')
        else:
            ax.scatter(t_raws[i] * 1e6, pop_raws[i], color=f'C{i+1}', s=25,
                    label=f'RID: {rids[i]}')
        

    ax.set_xlabel('Pulse duration [us]')
    ax.set_ylabel('Excited state population')
    ax.set_title(f'689 nm Rabi')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


    # # Theory curve: coupling factor applied, damped envelope included
    # C0           = get_coupling_factor(eps_0, quant_axis, mJ_targets[0])
    # Omega_theory = C0 * gamma_689 * np.sqrt(P_689 * 100 / (PI*w0**2) / Is_689)
    # Omega_eff    = np.sqrt(Omega_theory**2 + detunings[0]**2)

    # Readout model
    # avg_pop_meas     = apply_readout(avg_pop,     t_push)
    # avg_pop_meas_aom     = apply_readout(avg_pop_aom,     t_push)


    # I_peak          = 2 * P_689 * 100 / (PI * w0**2)           # peak intensity [uW/cm^2]
    # Omega_bare      = gamma_689 * np.sqrt(I_peak / (2 * Is_689)) # Rabi freq before coupling factor
    # T_rabi          = 2*PI / Omega_eff                           # Rabi period [s]
    # peak_excitation = Omega_theory**2 / Omega_eff**2             # on-resonance max population (no decay)
    # readout_fidelity = np.exp(-gamma_689 * t_push)

    




    # print("=" * 55)
    # print(f"Zeeman shift (mJ={mJ_targets[0]:+d}): "
    #       f"2pi x {delta_zeeman_689/(2*PI)*1e-3:.1f} kHz at B = {B_field_G:.1f} G")
    # print("  BEAM & POLARIZATION")
    # print(f"    Peak intensity (beam center):  {I_peak:.1f} uW/cm^2")
    # print(f"    Saturation intensity (Is_689): {Is_689:.2f} uW/cm^2")
    # print(f"    Saturation parameter  s=I/Is:  {I_peak/Is_689:.1f}")
    # print(f"    Coupling factor C = |eps_q|:   {C0:.4f}  "
    #       f"(mJ={mJ_targets[0]:+d})")
    # print()
    # print("  RABI DYNAMICS")
    # print(f"    Bare Rabi freq  (C=1):         2pi x {Omega_bare/(2*PI)*1e-6:.3f} MHz")
    # print(f"    Rabi freq       (with C):      2pi x {Omega_theory/(2*PI)*1e-6:.3f} MHz")
    # print(f"    Zeeman shift    (mJ={mJ_targets[0]:+d}):      2pi x {delta_zeeman_689/(2*PI)*1e-3:.1f} kHz  at B={B_field_G:.1f} G")
    # print(f"    Laser freq offset from bare:   2pi x {(detunings[0]+delta_zeeman_689)/(2*PI)*1e-3:.1f} kHz  [lab tuning ref, not in sim]")
    # print(f"    Sim detuning from mJ={mJ_targets[0]:+d} level: 2pi x {detunings[0]/(2*PI)*1e-3:.1f} kHz  [detunings[0] -> enters Hamiltonian]")
    # print(f"    Effective Rabi freq:           2pi x {Omega_eff/(2*PI)*1e-6:.3f} MHz")
    # print(f"    Rabi period:                   {T_rabi*1e6:.3f} us")
    # print(f"    pi-pulse time:                 {T_rabi/2*1e6:.3f} us")
    # print(f"    Peak excitation (no decay):    {peak_excitation:.4f}")
    # print()
    # print("  READOUT")
    # print(f"    Push duration:                 {t_push*1e6:.2f} us")
    # print(f"    Readout fidelity:              {readout_fidelity:.4f}  ({readout_fidelity:.1%})")
    # print(f"    Max observable population:     {peak_excitation * readout_fidelity:.4f}")
    # print("=" * 55)


