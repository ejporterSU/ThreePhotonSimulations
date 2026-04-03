#%%
"""
Sr-88 three-photon excitation: 1S0 -> 3P1(mJ=+1) -> 3S1(mJ=0) -> 3P0
Basic QuTiP Lindblad master equation structure.


"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.constants as const
from scipy.special import erf
from tqdm import tqdm


# --- HELPERS ----------------------
#region
def decompose_polarization(eps_hat, quant_axis):
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

def get_coupling_factor(eps_hat, quant_axis):
    """
    Compute the effective polarization coupling factor for driving a specific
    Zeeman sublevel.
    """
    eps_p1, eps_0, eps_m1 = decompose_polarization(eps_hat, quant_axis)
    return {+1: np.abs(eps_p1), 0: np.abs(eps_0), -1: np.abs(eps_m1)}


def get_zeeman_detuning(g_J, B_field):
    """
    Return the Zeeman frequency shift for a sublevel [rad/s].

        delta_Z = g_J * mu_B * B * mJ / hbar

    Args:
        g_J     (float): Lande g-factor.
        B_field (float): Bias magnetic field magnitude [T].

    Returns:
        float: Zeeman shift [rad/s]. Positive means the sublevel is shifted up in energy.
    """
    return g_J * MU_B * B_field / HBAR

#endregion

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

Ipref = H * C * np.pi / 3
Is_689 = Ipref * gamma_689 / (lambda_689**3) * 100  # [uW/cm^2]
Is_688 = Ipref * gamma_688 / (lambda_688**3) * 100  # [uW/cm^2]
Is_679 = Ipref * gamma_679 / (lambda_679**3) * 100  # [uW/cm^2]


# ── Basis states ──────────────────────────────────────────────────────────── #
#region
N  = 9
g = qt.basis(N, 0)   # |1S0>
e = qt.basis(N, 1)   # |3P1, mJ=+1>
v = qt.basis(N, 2)   # |3S1, mJ=0>
r = qt.basis(N, 3)   # |3P0>

states = [g, e, v, r]
projs = [state * state.dag() for state in states]
#endregion

#---- experimental parameters ------------------------------------------
#region
pol_vecs = (
    np.array([0.0, 0.0,  1.0]),   # 689 nm: z-hat
    np.array([0.0, 0.0,  1.0]),   # 688 nm: z-hat
    np.array([-1.0, 0.0,  0.0]),   # 679 nm: -x-hat (parallel to quant axis → pure pi)
)
quant_axis = np.array([-1.0, 0.0, 0.0])          # B-field along -x

theta_0,  theta_0z = np.radians(59.4384), 0.0   # 689 nm
theta_1,  theta_1z = np.radians(-59.64),  0.0   # 688 nm
theta_2,  theta_2z = 0.0,                 0.0   # 679 nm (along x)


w0_689       = 0.54e-3       # 689 nm 1/e^2 radius [m]
w0_688       = 0.90e-3       # 688 nm 1/e^2 radius [m]
w0_679       = 0.90e-3       # 679 nm 1/e^2 radius [m]
beam_radii = np.array([w0_689, w0_688, w0_679])

P_689        = 18.88e-3             # 689 nm peak power [W]
P_679        = 4.86e-3             # 679 nm peak power [W] 
P_688        = 8.77e-3             # 688 nm peak power [W] 

I_689 = 2*P_689/(PI*w0_689**2) * 100
I_688 = 2*P_688/(PI*w0_688**2) * 100
I_679 = 2*P_679/(PI*w0_679**2) * 100

# print(f"Powers \n689: {P_689*1e3:.2f} mW\n688: {P_688*1e3:.2f} mW\n679: {P_679*1e3:.2f} mW")

# Rabi frequencies [rad/s] 
Omega_689 = gamma_689 * np.sqrt(I_689/(2*Is_689))     # 689 nm:  1S0  <-> 3P1(mJ=+1)
Omega_688 = gamma_688 * np.sqrt(I_688/(2*Is_688))     # 688 nm:  3P1  <-> 3S1(mJ=0)
Omega_679 = gamma_679 * np.sqrt(I_679/(2*Is_679))     # 679 nm:  3S1  <-> 3P0

# print(f"O689 \n689: {Omega_689 / (2*PI*1e6):.2f} MHz\nO688: {Omega_688/ (2*PI*1e6):.2f} MHz\nO679: {Omega_679/ (2*PI*1e6):.2f} MHz")

B_field_G = 20
B_field_T = B_field_G * 1e-4
dwB_3p1= get_zeeman_detuning(G_J_3P1, B_field_T)


# print(f"{dwB_3p1 / (2*PI *1e6):.2f}, MHz Zeeman Shift")




# Single-photon detunings from each resonance [rad/s] 
delta_AC = 2*PI * 0.65e6
Delta_1 = dwB_3p1 + 2*PI * 5.0e6   # 689 nm detuning from 1S0 -> 3P1
Delta_2 = 2*PI *  400e6   # 688 nm detuning from 3P1 -> 3S1
Delta_3 = 2*PI *  400e6  + Delta_1  + delta_AC # 679 nm detuning from 3S1 -> 3P0

# ── Drive envelope ────────────────────────────────────────────────────────── #
# USE_RAMP = True  → erf ramp, reaches ~99% at t ≈ 2*tau_ramp (~200 ns total)
# USE_RAMP = False → square wave (instant turn-on)
USE_RAMP  = True
tau_ramp  = 90e-9   # erf width [s]; rise/fall spans 0→2*tau_ramp on each edge
T_MAX  = 3e-6   # total time [s]
dt = 10e-9
N_t    = int(T_MAX/dt) + 1
tlist  = np.linspace(0, T_MAX, N_t)


def drive_envelope(t, args):
    if USE_RAMP:
        # erf centered at tau_ramp: ~0 at t=0, ~1 at t=2*tau_ramp
        ramp_up   = 0.5 * (1.0 + erf(1.5 * t / tau_ramp - 1.5))
        # mirror at T_MAX: ~1 until T_MAX - 2*tau_ramp, ~0 at t=T_MAX
        ramp_down = 0.5 * (1.0 + erf(1.5 * (T_MAX - t) / tau_ramp - 1.5))
        return ramp_up * ramp_down
    return 1.0


# ── Hamiltonian in the rotating frame (RWA) ───────────────────────────────── #
# Energy levels shifted by accumulated detunings (ground state = 0)
couplings_689 = get_coupling_factor(pol_vecs[0], quant_axis)
couplings_688 = get_coupling_factor(pol_vecs[1], quant_axis)
couplings_679 = get_coupling_factor(pol_vecs[2], quant_axis)

MODE='TIME'


if MODE=='TIME':
    H_diag = (
        - (Delta_1 - dwB_3p1           )  * projs[1]
        - (Delta_1 + Delta_2           )  * projs[2]
        - (Delta_1 + Delta_2 - Delta_3 )  * projs[3]  
    )
   

    H_689 = Omega_689/2 * (couplings_689[+1] * (e * g.dag() + g * e.dag()) )
    H_688 = Omega_688/2 * ( couplings_688[-1] * (e * v.dag() + v * e.dag()) )
    H_679 = Omega_679/2 * (couplings_679[0] * (v * r.dag() + r * v.dag()))


    H_coupling = H_688 + H_679 + H_689
    if USE_RAMP:
        H = [H_diag, [H_coupling, drive_envelope]]
    else:
        H = H_diag + H_coupling

    # ── Collapse operators (Lindblad spontaneous emission) ────────────────────── #
    c_3P1_to_1S0 = [np.sqrt(gamma_689) * (g * e.dag())]
    c_3S1_to_3P1 = [np.sqrt(gamma_688) * (e * v.dag())]
    c_3S1_to_3P0 = [np.sqrt(gamma_679) * (r * v.dag())]
    c_ops = c_3P1_to_1S0 + c_3S1_to_3P1 + c_3S1_to_3P0 


    # ── Initial state ─────────────────────────────────────────────────────────── #
    rho0 = g*g.dag()   # all population in 1S0

    # ── Time evolution ────────────────────────────────────────────────────────── #


    result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs)
    pops   = result.expect   

    # ── Plot ──────────────────────────────────────────────────────────────────── #
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['g', 'e','v', 'r']
    colors = [f'C{i}' for i in range(4)]
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

    ac_starks = 2*PI*1e6*np.linspace(0.5,1, 10)
    max_clock = []
    for dac in ac_starks:
        H_diag = (
            - (Delta_1 - dwB_3p1           )  * projs[1]
            - (Delta_1 + Delta_2           )  * projs[2]
            - (Delta_1 + Delta_2 - (Delta_3 + dac) )  * projs[3]  
        )
        # Coherent couplings (Omega/2 for each laser field)
        H_689 = Omega_689/2 * (couplings_689[+1] * (e * g.dag() + g * e.dag()) )
        H_688 = Omega_688/2 * ( couplings_688[-1] * (e * v2.dag() + v * e.dag()) )
        H_679 = Omega_679/2 * (couplings_679[0] * (v * r.dag() + r * v.dag()))


        H_coupling = H_688 + H_679 + H_689
        if USE_RAMP:
            H = [H_diag, [H_coupling, drive_envelope]]
        else:
            H = H_diag + H_coupling

        # ── Collapse operators (Lindblad spontaneous emission) ────────────────────── #
        c_3P1_to_1S0 = [np.sqrt(gamma_689) * (g * e.dag()) ]
        c_3S1_to_3P1 = [np.sqrt(gamma_688) * (e * v.dag())]
        c_3S1_to_3P0 = [np.sqrt(gamma_679) * (r * v.dag())]
        c_ops = c_3P1_to_1S0 + c_3S1_to_3P1 + c_3S1_to_3P0 


        # ── Initial state ─────────────────────────────────────────────────────────── #
        rho0 = g*g.dag()   # all population in 1S0

        # ── Time evolution ────────────────────────────────────────────────────────── #



        result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs[-1])
        pops   = result.expect
        max_clock.append(max(pops[0]))

    plt.plot(ac_starks / (2*PI*1e6), max_clock)
    plt.xlabel("AC Stark [MHz]")
    plt.ylabel("Max Population")


