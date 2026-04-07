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
from scipy.optimize import curve_fit
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
e1 = qt.basis(N, 1)   # |3P1, mJ=-1>
e2 = qt.basis(N, 2)   # |3P1, mJ=0>
e3 = qt.basis(N, 3)   # |3P1, mJ=+1>
v1 = qt.basis(N, 4)   # |3S1, mJ=-1>
v2 = qt.basis(N, 5)   # |3S1, mJ=0>
v3 = qt.basis(N, 6)   # |3S1, mJ=+1>
r = qt.basis(N, 7)   # |3P0>
ds = qt.basis(N, 8)   # |3P2, mJ=all> dark state decay channel

states = [g, e1, e2, e3, v1, v2, v3, r, ds]
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

P_689        = 12.88e-3             # 689 nm peak power [W]
P_688        = 4.86e-3             # 688 nm peak power [W] 
P_679        = 8.77e-3             # 679 nm peak power [W] 


I_689 = 2*P_689/(PI*w0_689**2) * 100
I_688 = 2*P_688/(PI*w0_688**2) * 100
I_679 = 2*P_679/(PI*w0_679**2) * 100

# print(f"Powers \n689: {P_689*1e3:.2f} mW\n688: {P_688*1e3:.2f} mW\n679: {P_679*1e3:.2f} mW")

# Rabi frequencies [rad/s] 
Omega_689 = gamma_689 * np.sqrt(I_689/(2*Is_689))     # 689 nm:  1S0  <-> 3P1(mJ=+1)
Omega_688 = gamma_688 * np.sqrt(I_688/(2*Is_688))    # 688 nm:  3P1  <-> 3S1(mJ=0)
Omega_679 = gamma_679 * np.sqrt(I_679/(2*Is_679))  # 679 nm:  3S1  <-> 3P0

print(f"O689 \n689: {Omega_689/2 / (2*PI*1e6):.2f} MHz\nO688: {Omega_688/2/ (2*PI*1e6):.2f} MHz\nO679: {Omega_679/ (2*PI*1e6):.2f} MHz")


B_field_G = 20
B_field_T = B_field_G * 1e-4
dwB_3p1= get_zeeman_detuning(G_J_3P1, B_field_T)
dwB_3s1= get_zeeman_detuning(G_J_3S1, B_field_T)

# print(f"{dwB_3p1 / (2*PI *1e6):.2f}, MHz Zeeman Shift")




# Single-photon detunings from each resonance [rad/s] 
delta_AC = 2*PI * -1.1393e6
Delta_1 = dwB_3p1 + 2*PI * 5.0e6   # 689 nm detuning from 1S0 -> 3P1
Delta_2 = 2*PI *  -448e6   # 688 nm detuning from 3P1 -> 3S1
Delta_3 = 2*PI *  448e6  - Delta_1  # 679 nm detuning from 3S1 -> 3P0

# ── Drive envelope ────────────────────────────────────────────────────────── #
# USE_RAMP = True  → erf ramp, reaches ~99% at t ≈ 2*tau_ramp (~200 ns total)
# USE_RAMP = False → square wave (instant turn-on)
USE_RAMP  = True
tau_ramp  = 200e-9   # erf width [s]; rise/fall spans 0→2*tau_ramp on each edge
T_MAX  = 3e-6   # total time [s]
dt = 100e-9
N_t    = int(T_MAX/dt) + 1
tlist  = np.linspace(0, T_MAX, N_t)


def drive_envelope(t, args):
    if USE_RAMP:
        ramp_up   = 0.5 * (1.0 + erf(1.5 * t / tau_ramp - 1.5))
        ramp_down = 0.5 * (1.0 + erf(1.5 * (T_MAX - t) / tau_ramp - 1.5))
        return ramp_up * ramp_down
    return 1.0


# ── Hamiltonian in the rotating frame (RWA) ───────────────────────────────── #
# Energy levels shifted by accumulated detunings (ground state = 0)
couplings_689 = get_coupling_factor(pol_vecs[0], quant_axis)
couplings_688 = get_coupling_factor(pol_vecs[1], quant_axis)
couplings_679 = get_coupling_factor(pol_vecs[2], quant_axis)


H_diag = (
        - (Delta_1 + dwB_3p1           )  * projs[1]
        - (Delta_1                     )  * projs[2]
        - (Delta_1 - dwB_3p1           )  * projs[3]
        - (Delta_1 + Delta_2 + dwB_3s1 )  * projs[4]
        - (Delta_1 + Delta_2           )  * projs[5]
        - (Delta_1 + Delta_2 - dwB_3s1 )  * projs[6]
        - (Delta_1 + Delta_2 + Delta_3 + delta_AC )  * projs[7]  
    )   



H_689 = Omega_689/2 * (couplings_689[-1] * (e1 * g.dag() + g * e1.dag()) + 
                    couplings_689[+1] * (e3 * g.dag() + g * e3.dag()) )

H_688 = Omega_688/2 * (couplings_688[+1] * (e1 * v2.dag() + v2 * e1.dag()) + 
                        couplings_688[-1] * (e2 * v1.dag() + v1 * e2.dag()) +
                        couplings_688[+1] * (e2 * v3.dag() + v3 * e2.dag()) + 
                        couplings_688[-1] * (e3 * v2.dag() + v2 * e3.dag()) )

H_679 = Omega_679/2 * (couplings_679[0] * (v2 * r.dag() + r * v2.dag()))


def get_AC_stark(omega1, dwb, omega3, delta1, delta3):
    delta1 = np.abs(delta1)
    delta3 = np.abs(delta3)
    # Ground state shift from 689 (blue detuned from both mJ=±1 sublevels):
    #   Each sublevel has coupling factor 1/sqrt(2) (z-pol ⊥ quant axis),
    #   so Omega_eff^2 = (1/sqrt(2))^2 * omega1^2 = omega1^2/2
    #   delta_g = omega1^2/2 * delta1/(4*(delta1^2 - dwb^2)/2)... simplified:
    #   delta_g = +omega1^2 * delta1 / (4*(delta1^2 - dwb^2))
    # Clock state shift from 679 (|3P0> is BELOW |3S1>, so shift is negative):
    #   delta_r = -omega3^2 / (4*delta3)
    delta_g = omega1**2 * delta1 / (4 * (delta1**2 - dwb**2))
    delta_r = -omega3**2 / (4 * delta3)
    return delta_r - delta_g  # both terms negative

stark_est = get_AC_stark(Omega_689, dwB_3p1, Omega_679, Delta_1, Delta_3)
print(f"AC Stark est: {stark_est / (2*PI * 1e6):.3f} MHz" )


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

if USE_RAMP:
    H = [H_diag, [H_coupling, drive_envelope]]
else:
    H = H_diag + H_coupling

# ── Initial state ─────────────────────────────────────────────────────────── #
rho0 = g*g.dag()   # all population in 1S0


MODE='TIME'


if MODE=='TIME':
    # ── Time evolution ────────────────────────────────────────────────────────── #
    result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs)
    pops   = result.expect   

    # ── Plot ──────────────────────────────────────────────────────────────────── #
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['g', 'em1', 'e0', 'e1', 'vm1', 'v0', 'v1', 'r', 'ds']
    colors = [f'C{i}' for i in range(9)]
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
    ac_starks = 2*PI*1e6*np.linspace(-2.5,  0, 10)
    max_clock = []
    
    for dac in tqdm(ac_starks, desc='Running AC Stark Sweep...'):
        H_diag = (
                - (Delta_1 + dwB_3p1           )  * projs[1]
                - (Delta_1                     )  * projs[2]
                - (Delta_1 - dwB_3p1           )  * projs[3]
                - (Delta_1 + Delta_2 + dwB_3s1 )  * projs[4]
                - (Delta_1 + Delta_2           )  * projs[5]
                - (Delta_1 + Delta_2 - dwB_3s1 )  * projs[6]
                - (Delta_1 + Delta_2 + Delta_3 + dac)  * projs[7]
            )
        if USE_RAMP:
            H = [H_diag, [H_coupling, drive_envelope]]
        else:
            H = H_diag + H_coupling
        result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=projs[-2])
        max_clock.append(max(result.expect[0]))

    x_data = ac_starks / (2*PI*1e6)
    y_data = np.abs(np.array(max_clock))

    # Fit a Lorentzian to extract the AC Stark shift from the peak position
    def lorentzian(x, x0, A, gamma, offset):
        return A * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + offset

    idx_peak = np.argmax(y_data)
    p0 = [x_data[idx_peak], y_data[idx_peak] ,
          (x_data[-1] - x_data[0]) / 4, 0]

    popt, pcov = curve_fit(lorentzian, x_data, y_data, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    x0_fit, A_fit, gamma_fit, offset_fit = popt
    x0_err = perr[0]

    x_fine = np.linspace(x_data[0], x_data[-1], 500)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_data, y_data, 'o', label='Simulation')
    ax.plot(x_fine, lorentzian(x_fine, *popt), '-', label='Lorentzian fit')
    ax.axvline(x0_fit, color='r', linestyle='--', label=f'Peak: {x0_fit:.4f} MHz')
    ax.set_xlabel("AC Stark [MHz]")
    ax.set_ylabel("Max Population")
    ax.set_title(f"AC Stark Shift = {x0_fit:.4f} ± {x0_err:.4f} MHz")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"AC Stark shift (Lorentzian peak): {x0_fit:.4f} ± {x0_err:.4f} MHz")
    print(f"Linewidth (FWHM):                 {abs(gamma_fit):.4f} MHz")



# %%

omega3 = np.array([32.27, 38.94, 49.66, 60.44, 18.87])
ac3 = np.array([-1.385, -1.65, -2.17, -2.84, -1.002])

omega1 = np.array([4.39, 3.33, 2.45, 1.77, 5.54])
ac1 = np.array([-1.385, -1.06, -0.85, -0.71, -1.78])

x_fit = omega1**2
ac = ac1



coeffs, cov = np.polyfit(x_fit, ac, 1, cov=True)
slope, intercept = coeffs
slope_err, intercept_err = np.sqrt(np.diag(cov))

x_line = np.linspace(x_fit.min(), x_fit.max(), 300)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x_fit, ac, zorder=3, label='Data')
ax.plot(x_line, np.polyval(coeffs, x_line), 'r-',
        label=f'Linear fit: slope = {slope*1e3:.4f} ± {slope_err*1e3:.4f} kHz/(MHz)²')
ax.set_xlabel("Ω₃² [(MHz)²]")
ax.set_ylabel("AC Stark shift [MHz]")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Slope:     {slope:.4e} ± {slope_err:.4e} MHz/(rad/s)²")
print(f"Intercept: {intercept:.4f} ± {intercept_err:.4f} MHz")