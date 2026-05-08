#%%
"""
Sr-88 three-photon excitation: 1S0 -> 3P1(mJ=+1) -> 3S1(mJ=0) -> 3P0
Basic QuTiP Lindblad master equation structure.

State ordering:
|0> = 1S0            (ground)
|1> = 3P1, mJ=-1     (intermediate 1, driven by 689 nm sigma-)
|2> = 3P1, mJ=0      (intermediate 1, driven by 689 nm pi)
|3> = 3P1, mJ=+1     (intermediate 1, driven by 688 nm sigma+)
|4> = 3S1, mJ=-1     (intermediate 2, driven by 688 nm sigma-)
|5> = 3S1, mJ=0      (intermediate 2, driven by 688 nm pi)
|6> = 3S1, mJ=+1     (intermediate 2, driven by 688 nm simga+)
|7> = 3P0            (clock state)
|8> = 3P2, all mJ    (dark decay channel 3P2)

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

# --- Atomic Parameters ---------------------------------------
#region
# lande-g factors
G_J_1S0 = 0
G_J_3P0 = 0 
G_J_3P1 = 1.50116
G_J_3P2 = 1.50116
G_J_3S1 = 2.00232

gamma_689, lambda_689 = 2*PI * 7.48e3,  689.4489e-9
gamma_688, lambda_688 = 2*PI * 3.90e6,  688.020770e-9
gamma_679, lambda_679 = 2*PI * 1.26e6,  679.288943e-9
gamma_707, lambda_707 = 2*PI * 6.225e6, 707.197215e-9

# ── Basis states ──────────────────────────────────────────────────────────── #
#region
N  = 9  
g = qt.basis(N, 0)   # |1S0>
e1 = qt.basis(N, 1)   # |3P1, mJ=-1>
e2 = qt.basis(N, 2)   # |3P1, mJ=0>
e3 = qt.basis(N, 3)   # |3P1, mJ=+1>
v1 = qt.basis(N, 4)   # |3S1, mJ=-1>
v2 = qt.basis(N, 5)   # |3S1, mJ=0>
v3 = qt.basis(N, 6)   # |3S1, mJ=+1>
ds = qt.basis(N, 7)   # |3P2, mj=all>
r = qt.basis(N, 8)   # |3P0> 

states = [g, e1, e2, e3, v1, v2, v3, ds, r]
projs = [state * state.dag() for state in states]
#endregion

#---- experimental parameters ------------------------------------------
#region

theta_0,  theta_0z = np.radians(59.4384), 0.0   # 689 nm
theta_1,  theta_1z = np.radians(-59.64),  0.0   # 688 nm
theta_2,  theta_2z = 0.0,                 0.0   # 679 nm (along x)

w0_689       = 0.54e-3       # 689 nm 1/e^2 radius [m]
w0_688       = 0.90e-3       # 688 nm 1/e^2 radius [m]
w0_679       = 0.90e-3       # 679 nm 1/e^2 radius [m]
beam_radii = np.array([w0_689, w0_688, w0_679])

# Rabi frequencies [rad/s] 
Omega_689 =  2*PI * 3.75e6    # 689 nm:  1S0  <-> 3P1(mJ=+1)
Omega_688 =  2*PI * 21.6e6     # 688 nm:  3P1  <-> 3S1(mJ=0)
Omega_679 =  2*PI * 21.4e6     # 679 nm:  3S1  <-> 3P0
dwB_3p1 =    2*PI * 40e6
dwB_3s1 =    dwB_3p1 * 4/3 # coming from relative landau factor


# Single-photon detunings from each resonance [rad/s] 
delta_AC = 2*PI * -0.395e6
dwB_3p1  = 2*PI * 40e6
delta_1  = 2*PI * 45e6
delta_2  = 2*PI * 355e6

Delta_1 = delta_1   # 689 nm detuning from 1S0 -> 3P1
Delta_2 = delta_1 + delta_2   # 688 nm detuning from 3P1 -> 3S1
Delta_3 = delta_AC # 679 nm detuning from 3S1 -> 3P0


# ── sim params ────────────────────────────────────────────────────────── #
MODE='TIME'
USE_RAMP  = True
PLOT_ENVELOPE = True   # set True to preview the drive envelope before running
T_MAX  = 10e-6   # total time [s]
dt = 20e-9
N_t    = int(T_MAX/dt) + 1
tlist  = np.linspace(0, T_MAX, N_t)
ac_starks = 2*PI*1e6*np.linspace(-0.5, -0.3, 20)
tau_ramp  = 80e-9   # erf width [s]; rise/fall spans 0→2*tau_ramp on each edge
N_t_ramp = 50

def drive_envelope(t, args):
    if USE_RAMP:
        T_pulse = args.get('T_total', T_MAX)  # pulse window end [s]
        t_rise = 100e-9              # AOM turn-on delay: 50% rise point
        t_fall = T_pulse + 45e-9    # AOM turn-off offset: 50% fall point

        ramp_up   = 0.5 * (1.0 + erf((t - t_rise) / tau_ramp))
        ramp_down = 0.5 * (1.0 - erf((t - t_fall) / tau_ramp))
        return ramp_down * ramp_up
    return 1.0


if PLOT_ENVELOPE and USE_RAMP:
    preview_t_ons = [0, T_MAX / 4, T_MAX / 2, T_MAX * 3 / 4, T_MAX]
    fig_env, ax_env = plt.subplots(figsize=(7, 3))
    for t_on in preview_t_ons:
       
        T_sim   = t_on + 150e-9   # sim end: fall is complete here
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

if MODE=='TIME':
    H_diag = (
        - (Delta_1 + dwB_3p1           )  * projs[1]
        - (Delta_1                     )  * projs[2]
        - (Delta_1 - dwB_3p1           )  * projs[3]
        - (Delta_2 + dwB_3s1           )  * projs[4]
        - (Delta_2                     )  * projs[5]
        - (Delta_2 - dwB_3s1           )  * projs[6]
        - (0                           )  * projs[7] 
        - (Delta_3                     )  * projs[8]  
    )   


    # Coherent couplings (Omega/2 for each laser field)
    H_689 = Omega_689/2 * (e1*g.dag() + g*e1.dag() +
                           e3*g.dag() + g*e3.dag())

    H_688 = Omega_688/2 * ( e1 * v2.dag() + v2 * e1.dag() + 
                            e2 * v1.dag() + v1 * e2.dag() +
                            e2 * v3.dag() + v3 * e2.dag() + 
                            e3 * v2.dag() + v2 * e3.dag() )

    H_679 = Omega_679/2 * (v2 * r.dag() + r * v2.dag())
    H_coupling = H_688 + H_679 + H_689  

    # ── Collapse operators (Lindblad spontaneous emission) ────────────────────── #
    c_3P1_to_1S0 = [np.sqrt(gamma_689) * (g * e1.dag()),
                    np.sqrt(gamma_689) * (g * e2.dag()),
                    np.sqrt(gamma_689) * (g * e3.dag()) ]
    c_3S1_to_3P1 = [np.sqrt(gamma_688/2) * (e2 * v1.dag()), np.sqrt(gamma_688/2) * (e1 * v1.dag()),
                    np.sqrt(gamma_688/2) * (e1 * v2.dag()), np.sqrt(gamma_688/2) * (e3 * v2.dag()),
                    np.sqrt(gamma_688/2) * (e3 * v3.dag()), np.sqrt(gamma_688/2) * (e2 * v3.dag()), ]
    c_3S1_to_3P0 = [np.sqrt(gamma_679) * (r * v1.dag()), 
                    np.sqrt(gamma_679) * (r * v2.dag()),
                    np.sqrt(gamma_679) * (r * v3.dag())]
    c_3S1_to_3P2 = [np.sqrt(gamma_707) * (ds * v1.dag()), 
                    np.sqrt(gamma_707) * (ds * v2.dag()),
                    np.sqrt(gamma_707) * (ds * v3.dag())]
    c_ops = c_3P1_to_1S0 + c_3S1_to_3P1 + c_3S1_to_3P0 + c_3S1_to_3P2

    rho0 = g*g.dag()

    if USE_RAMP:
        t_on_list = np.linspace(0, T_MAX, N_t_ramp)

        H = [H_diag, [H_coupling, drive_envelope]]
        final_pops = [[] for _ in range(len(states))]

        for t_on in tqdm(t_on_list, desc="Scanning on-time..."):
            T_sim   = t_on + 150e-9   # sim end: fall is complete here
            n_pts = max(int(T_sim / dt) + 1, 20)
            tlist_i = np.linspace(0, T_sim, n_pts)
            result = qt.mesolve(H, rho0, tlist_i, c_ops, e_ops=projs,
                                args={'T_total': t_on})
            for i, pop in enumerate(result.expect):
                final_pops[i].append(pop[-1])



        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['g', 'em1', 'e0', 'e1', 'vm1', 'v0', 'v1', 'ds', 'r']
        colors = ['C0', 'C1', 'k', 'C2', 'k', 'C3', 'k', 'C4', 'C5']
        for pop, label, color in zip(final_pops, labels, colors):
            ax.plot(t_on_list * 1e6, pop, color=color, label=f"{label} {max(np.abs(pop)):.3f}")
            ax.scatter(t_on_list * 1e6, pop, color=color)
        ax.set_xlabel('Time [µs]')
        ax.set_ylabel('Final Population')
        ax.set_title('Sr-88 Three Photon Simulation')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:
        H = H_diag + H_coupling
        result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs)
        pops   = result.expect

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['g', 'em1', 'e0', 'e1', 'vm1', 'v0', 'v1', 'ds', 'r']
        colors = ['C0', 'C1', 'k', 'C2', 'k', 'C3', 'k', 'C4', 'C5']
        for pop, label, color in zip(pops, labels, colors):
            ax.plot(tlist * 1e6, pop, color=color, label=f"{label} {max(np.abs(pop)):.3f}")
        ax.set_xlabel('Time [µs]')
        ax.set_ylabel('Population')
        ax.set_title('Sr-88 Three Photon Simulation')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.show()

if MODE=='FREQ':
    
    max_clock = []
    for dac in tqdm(ac_starks, desc="Running Stark Scan..."):
        H_diag = (
            - (Delta_1 + dwB_3p1           )  * projs[1]
            - (Delta_1                     )  * projs[2]
            - (Delta_1 - dwB_3p1           )  * projs[3]
            - (Delta_2 + dwB_3s1           )  * projs[4]
            - (Delta_2                     )  * projs[5]
            - (Delta_2 - dwB_3s1           )  * projs[6]
            - (0                           )  * projs[7] 
            - (dac                     )  * projs[8]  
        )   


        # Coherent couplings (Omega/2 for each laser field)
        H_689 = Omega_689/2 * (e1*g.dag() + g*e1.dag() +
                            e3*g.dag() + g*e3.dag())

        H_688 = Omega_688/2 * ( e1 * v2.dag() + v2 * e1.dag() + 
                                e2 * v1.dag() + v1 * e2.dag() +
                                e2 * v3.dag() + v3 * e2.dag() + 
                                e3 * v2.dag() + v2 * e3.dag() )

        H_679 = Omega_679/2 * (v2 * r.dag() + r * v2.dag())
        H_coupling = H_688 + H_679 + H_689  

        # ── Collapse operators (Lindblad spontaneous emission) ────────────────────── #
        c_3P1_to_1S0 = [np.sqrt(gamma_689) * (g * e1.dag()),
                        np.sqrt(gamma_689) * (g * e2.dag()),
                        np.sqrt(gamma_689) * (g * e3.dag()) ]
        c_3S1_to_3P1 = [np.sqrt(gamma_688/2) * (e2 * v1.dag()), np.sqrt(gamma_688/2) * (e1 * v1.dag()),
                        np.sqrt(gamma_688/2) * (e1 * v2.dag()), np.sqrt(gamma_688/2) * (e3 * v2.dag()),
                        np.sqrt(gamma_688/2) * (e3 * v3.dag()), np.sqrt(gamma_688/2) * (e2 * v3.dag()), ]
        c_3S1_to_3P0 = [np.sqrt(gamma_679) * (r * v1.dag()), 
                        np.sqrt(gamma_679) * (r * v2.dag()),
                        np.sqrt(gamma_679) * (r * v3.dag())]
        c_3S1_to_3P2 = [np.sqrt(gamma_707) * (ds * v1.dag()), 
                        np.sqrt(gamma_707) * (ds * v2.dag()),
                        np.sqrt(gamma_707) * (ds * v3.dag())]
        c_ops = c_3P1_to_1S0 + c_3S1_to_3P1 + c_3S1_to_3P0 + c_3S1_to_3P2


        # ── Initial state ─────────────────────────────────────────────────────────── #
        rho0 = g*g.dag()   # all population in 1S0

        # ── Time evolution ────────────────────────────────────────────────────────── #


        result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs[-1])
        pops   = result.expect 
        max_clock.append(max(pops[0]))

    plt.plot(ac_starks / (2*PI*1e6), max_clock)

    imax = np.argmax(max_clock)
    plt.axvline(ac_starks[imax]/(2*PI*1e6), label=f"Peak: {ac_starks[imax]/(2*PI*1e6):0.3f} MHz")

    plt.legend()
    plt.xlabel("AC Stark [MHz]")
    plt.ylabel("Max Population")









# %%
