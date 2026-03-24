"""
aom_rabi_sim.py
---------------
Simple two-level system simulation driven by a shaped optical pulse.
Edit the PARAMETERS block and PULSE_SHAPE function at the top, then run.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib.pyplot as plt

# =============================================================================

# PARAMETERS
# =============================================================================

Omega_0   = 2 * np.pi * 4e6     # peak Rabi frequency (rad/s)
delta     = 2 * np.pi * 5e6     # detuning (rad/s)
T         = 1000e-9               # total pulse duration (s)

# AOM rise time (beam waist / sound speed)
beam_waist = 500e-6              # m
v_sound    = 4.2e3               # m/s  (TeO2)
tau_rise   = beam_waist / v_sound

# =============================================================================
# PULSE SHAPE  —  edit this function
# Returns Omega(t) given t (scalar or array), using globals above.
# =============================================================================

def pulse_shape(t, pulse_T=None):
    """
    Smooth erf-edge pulse: rises over ~tau_rise, flat top at Omega_0, falls over ~tau_rise.
    sigma = tau_rise / 3 so the erf is ~99% complete within tau_rise.
    """
    if pulse_T is None:
        pulse_T = T
    t = np.asarray(t, dtype=float)
    sigma = tau_rise / 3.0
    rise = 0.5 * (1 + erf((t - tau_rise) / (np.sqrt(2) * sigma)))
    fall = 0.5 * (1 - erf((t - (pulse_T - tau_rise)) / (np.sqrt(2) * sigma)))
    return Omega_0 * rise * fall

# Sweep range for pulse duration
T_min = 10e-9    # s
T_max = 1000e-9   # s
N_T   = 150

# =============================================================================
# SOLVER
# =============================================================================

def solve_excitation(pulse_T):
    """Integrate TDSE for a pulse of duration pulse_T, return final |c_e|^2."""
    def rhs(t, y):
        Omega_t = float(pulse_shape(np.array([t]), pulse_T)[0])
        c_g = y[0] + 1j * y[1]
        c_e = y[2] + 1j * y[3]
        dc_g = -1j * 0.5 * (-delta * c_g + Omega_t * c_e)
        dc_e = -1j * 0.5 * ( Omega_t * c_g + delta  * c_e)
        return [dc_g.real, dc_g.imag, dc_e.real, dc_e.imag]

    sol = solve_ivp(rhs, (0, pulse_T), [1, 0, 0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    t_eval=np.linspace(0, pulse_T, 500))
    c_e = sol.y[2, -1] + 1j * sol.y[3, -1]
    return float(np.abs(c_e)**2)


T_arr   = np.linspace(T_min, T_max, N_T)
Pe_sim  = np.array([solve_excitation(Ti) for Ti in T_arr])

# Analytic square-pulse prediction
Omega_eff       = np.sqrt(Omega_0**2 + delta**2)
Pe_analytic     = (Omega_0 / Omega_eff)**2 * np.sin(Omega_eff * T_arr / 2)**2

# =============================================================================
# PLOT
# =============================================================================

t_ex = np.linspace(0, T, 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
fig.suptitle(
    fr"Two-level system  |  $\Omega_0 = 2\pi \times {Omega_0/(2*np.pi*1e6):.1f}$ MHz"
    fr",  $\delta = 2\pi \times {delta/(2*np.pi*1e6):.1f}$ MHz"
    fr",  $\tau_\mathrm{{rise}} = {tau_rise*1e9:.1f}$ ns"
)

# Panel 1: example pulse shape at the reference T
ax1.plot(t_ex * 1e9, pulse_shape(t_ex) / (2 * np.pi * 1e6), color='tab:blue')
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel(r"$\Omega(t)\,/\,2\pi$ (MHz)")
ax1.set_title(f"Pulse shape example  (T = {T*1e9:.0f} ns)")

# Panel 2: excitation vs pulse duration
ax2.plot(T_arr * 1e9, Pe_sim,      color='tab:orange', lw=2,   label='Simulation (smooth pulse)')
ax2.plot(T_arr * 1e9, Pe_analytic, color='black',      lw=1.5, ls='--', label='Analytic square pulse')
ax2.set_xlabel("Pulse Duration T (ns)")
ax2.set_ylabel(r"Final Excitation  $|c_e|^2$")
ax2.set_title("Excitation vs Pulse Duration")
ax2.legend()
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
