#%%
import numpy as np
import matplotlib.pyplot as plt
import time
from simulation_functions import *

def pd_mv_to_power(beam, pd_mv):
    """
     converts photodiode read value to output power in W
     for each of the three beam options
    """
    if beam == '689':
        return 1e-3*(0.229*pd_mv - 0.586 ) 
    elif beam == '688':
        return 1e-3*(0.133*pd_mv + 0.471 )
    elif beam == '679':
        return 1e-3*(0.146*pd_mv + 0.008 )
    else:
        raise Exception("Not valid beam")

# ── Fixed geometry ───────────────────────────────────────────────────────── #
# Beam propagation angles
theta_0,  theta_0z = np.radians(59.4384), 0.0   # 689 nm
theta_1,  theta_1z = np.radians(-59.64),  0.0   # 688 nm
theta_2,  theta_2z = 0.0,                 0.0   # 679 nm (along x)

k_vec_0 = (2*PI / lambda_689) * get_k_hat(theta_0, theta_0z)
k_vec_1 = (2*PI / lambda_688) * get_k_hat(theta_1, theta_1z)
k_vec_2 = (2*PI / lambda_679) * get_k_hat(theta_2, theta_2z)
k_vecs  = (k_vec_0, k_vec_1, k_vec_2)


quant_axis = np.array([-1.0, 0.0, 0.0])          # B-field along -x

# Polarizations:
#   689 nm (z-hat): sigma+ component drives 1S0(mJ=0) → 3P1(mJ=+1)
#   688 nm (z-hat): sigma- component drives 3P1(mJ=+1) → 3S1(mJ=0)  (delta_mJ=-1)
#   679 nm (y-hat): pi component drives 3S1(mJ=0) → 3P0(mJ=0) (delta_mJ=0)
pol_vecs = (
    np.array([0.0, 0.0,  1.0]),   # 689 nm: z-hat
    np.array([0.0, 0.0,  1.0]),   # 688 nm: z-hat
    np.array([-1.0, 0.0,  0.0]),   # 679 nm: -x-hat (parallel to quant axis → pure pi)
)
# mJ_target = polarization selection (delta_mJ) for coupling factor:
#   +1 = sigma+,  0 = pi,  -1 = sigma-
mJ_targets = (+1, -1, 0 )

B_field_G = 20
B_field_T = B_field_G * 1e-4
delta_zeeman_689 = get_zeeman_detuning(G_J_3P1, mJ_targets[0], B_field_T)


# --- laser powers ---
# via PD
P_689        = pd_mv_to_power("689", 85)   # 689 nm peak power [W]
P_679        = pd_mv_to_power("679", 60)   # 679 nm peak power [W] 
P_688        = pd_mv_to_power("688", 33)   # 688 nm peak power [W] 
# direct measurement
# P_689        = 0.0e-3             # 689 nm peak power [W]
# P_679        = 0.0e-3             # 679 nm peak power [W] 
# P_688        = 0.0e-3             # 688 nm peak power [W] 

powers = [P_689, P_688, P_679]
print(f"Powers \n689: {P_689*1e3:.2f} mW\n688: {P_688*1e3:.2f} mW\n679: {P_679*1e3:.2f} mW")


# --- residual detunings  ---
detuning_mj1 = 2*PI * 5.0e6                            # detuning from 3p1, mj=1 <-> 1s0
detuning_689   = delta_zeeman_689 + detuning_mj1       # 689 nm detuning from 1S0→3P1(mJ=0) [rad/s]
detuning_679   = 2*PI * -400e6                         # 679 nm detuning from 3S1→3P0       [rad/s]
detunings = [detuning_mj1, 
             detuning_679 - detuning_mj1, 
             detuning_679]
print(rf"""Input Detunings:
    delta_0={detunings[0] * 1e-6 / (2*PI):.2f} MHz
    delta_1={detunings[1] * 1e-6 / (2*PI):.2f} MHz
    delta_2={detunings[2] * 1e-6 / (2*PI):.2f} MHz""")

# --- atomic cloud ---
cloud_radii  = np.array([0e-6, 0e-6, 0e-6])   # 1-sigma position widths [m]
temperatures = np.array([0e-6, 0e-6, 0e-6])   # thermal widths [K]

# --- beam waists ---
w0_689       = 0.54e-3       # 689 nm 1/e^2 radius [m]
w0_688       = 0.90e-3       # 688 nm 1/e^2 radius [m]
w0_679       = 0.90e-3       # 679 nm 1/e^2 radius [m]
beam_radii = np.array([w0_689, w0_688, w0_679])

# --- simulation ---
T_MAX   = 5e-6
dt      = 50e-9
N_atoms = 100
t_push = 0.8e-6
n_shots = 20
# --- misc params ---
ep      = {'t0': 0.0, 'sigma': 90e-9}
envelope='ERF'
mode="TIME" # "TIME" for time scan(normal rabi flopping)
            # "FREQ" for frequency scans to find stark shift
delta_AC = 2*PI * 1e6 * 1.35



### RUN CODE NOW
# Sample ensemble; atleast_2d ensures shape (N, 3) for vectorized functions
pos, vel = sample_atomic_ensemble(cloud_radii, temperatures, n_samples=N_atoms if mode=="TIME" else 1)
pos = np.atleast_2d(pos)
vel = np.atleast_2d(vel)

if mode == "TIME":
    detunings[2] = detunings[2] + delta_AC

    t0 = time.perf_counter()
    tlist, pops, params = simulate_three_photon_rabi_dynamics_new(
        pos, vel, beam_radii, powers, detunings, k_vecs,
        pol_vecs, quant_axis, mJ_targets,
        t_max=T_MAX, dt=dt,
        n_shots=n_shots,
        envelope=envelope,
        envelope_params=ep, n_jobs=-1
    )
    print(f"Timing: {time.perf_counter()-t0:.2f} s  "
          f"({n_shots} shots, dt={dt*1e9:.0f} ns, T_MAX={T_MAX*1e6:.1f} µs)")
    print_simulation_params(params)

    fig, ax = plt.subplots(figsize=(8, 4))
    state_labels = ['1S0', '3P1', '3P0', '3P2']
    colors       = ['C0', 'C1', 'C3', 'C4']
    for k in range(4):
        ax.scatter(tlist * 1e6, pops[k], color=colors[k], label=state_labels[k])
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('State population')
    ax.set_title(f'Three-photon Simultaneous Excitation — Rabi flopping\n'
                 rf'$\Delta_0$={detunings[0]/(2*PI)*1e-6:.1f} MHz, '
                 rf'$\Delta_1$={detunings[1]/(2*PI)*1e-6:.1f} MHz, '
                 rf'$\Delta_2$={detunings[2]/(2*PI)*1e-6:.1f} MHz')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()

elif mode == "FREQ":
    # Scan detunings[2] (679 nm) around the bare three-photon resonance to locate
    # the AC Stark-shifted resonance. All other detunings are held fixed.
    t_probe    = T_MAX   # fixed pulse time [s]; tune to ~quarter pi-time for contrast
    dfi = 2*PI * 1e6     # scan ±50 MHz around the bare resonance
    dff = 2*PI * 2e6
    n_points   = 21

    bare_resonance = detunings[2]   # = detunings[2] as currently set
    scan_d2 = np.linspace(bare_resonance + dfi,
                          bare_resonance + dff, n_points)

    pop_3P0 = np.zeros(n_points)
    for idx, d2 in enumerate(tqdm(scan_d2, desc='freq scan (679 detuning)')):
        dets_scan = [detunings[0], detunings[1], d2]
        _, pops_s, _ = simulate_three_photon_rabi_dynamics_new(
            pos, vel, beam_radii, powers, dets_scan, k_vecs,
            pol_vecs, quant_axis, mJ_targets,
            t_max=t_probe, dt=dt,
            envelope="SQUARE",
            envelope_params=ep,
        )
        pop_3P0[idx] = max(pops_s[2, :])   # 3P0 population (row 2: 1S0, 3P1, 3P0, 3P2)

    # x-axis: offset of detunings[2] from bare resonance [MHz]
    x_MHz     = (scan_d2 - bare_resonance) / (2*PI) * 1e-6
    peak_idx  = np.argmax(pop_3P0)
    stark_MHz = x_MHz[peak_idx]

    print(f"Simulated peak at Δ₂ offset = {stark_MHz:.4f} MHz  →  AC Stark shift ≈ {stark_MHz:.2f} MHz")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_MHz, pop_3P0, 'C3.-')
    ax.axvline(stark_MHz, color='C3', ls=':',  lw=1.0,
               label=f'peak @ {stark_MHz:.2f} MHz')
    ax.set_xlabel(r'$\Delta_2$ offset from bare resonance [MHz]')
    ax.set_ylabel('3P0 population')
    ax.set_title(f'679 nm frequency scan  (t_probe = {t_probe*1e6:.1f} µs)\n'
                 rf'$\Delta_0$={detunings[0]/(2*PI)*1e-6:.1f} MHz, '
                 rf'$\Delta_1$={detunings[1]/(2*PI)*1e-6:.1f} MHz')
    # ax.set_ylim(0, None)
    ax.legend()
    plt.tight_layout()
    plt.show()

else:
    print("Set mode to 'TIME' or 'FREQ'.")
