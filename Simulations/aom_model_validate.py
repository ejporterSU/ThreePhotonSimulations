import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_functions import erf_rabi_envelope

#%%
from scipy.special import erf, erfinv

def read_file(file_name, ind):
    data = np.loadtxt(f"Data/{file_name}_{ind}.txt",
                      delimiter='\t', skiprows=20)
    t, sig, ttl = data[:, 0], data[:, 1], data[:, 4]
    return t, sig, ttl


def eval_envelope(coeff_fn, t_array):
    """
    Evaluate a coeff function from erf_rabi_envelope on a time array.
    Handles the short-circuit case that returns a scalar 0.0 for very
    short pulses (t_pulse < 50 ns).
    """
    raw = coeff_fn(t_array)
    if np.isscalar(raw):
        return np.full_like(t_array, raw, dtype=float)
    return np.asarray(raw, dtype=float)


def test_erf_rabi_envelope(t0, sigma, t_pulse, Omega_peak=1.0):
    t_off = t0 + t_pulse - 55e-9  # 50% fall point
    t_on = t0 
    
    def f(t, _args=None):
        rise = 0.5 * (1.0 + erf((t-t_on)    / sigma))
        fall = 0.5 * (1.0 - erf((t - t_off) / (sigma*0.65)))
        intensity =  rise * fall
        return Omega_peak * np.sqrt(np.maximum(intensity, 0.0))


    params = dict(t0=t0, sigma=sigma, t_pulse=t_pulse, Omega_peak=Omega_peak)
    if t_pulse < 50e-9:  # AOM can't turn on this fast; returns zero for t_pulse < 50 ns
        return lambda t, _args=None: 0.0, params
    else:
        return f, params


# ── Parameters to match what the simulation uses ─────────────────────────── #
SIGMA_AOM   = 90e-9     # erf width [s]  (ep['sigma'] in simulate_one_photon)
T0_AOM      = 0.0       # 50% rise reference; here we'll set it per shot from TTL

PULSE_TIMES = np.linspace(0, 2e-6, 15)   # same as used in simulation

# ── Plot grid ─────────────────────────────────────────────────────────────── #
fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharey=True)
axes_flat  = axes.flatten()

for i, t_pulse in enumerate(PULSE_TIMES):
    ax = axes_flat[i]

    # --- Load data ---
    t, sig, ttl = read_file('rise_time', i)
    sig_norm = (sig*1e3 - 6) * 0.229 / 20
    sig_norm_rabi = np.sqrt((sig*1e3 - 6) * 0.229 / 20)


    # --- Build ERF model using erf_rabi_envelope from simulation_functions ---
    # Omega_peak=1 → coeff(t) = sqrt(intensity), so coeff(t)^2 = intensity
    coeff, params = test_erf_rabi_envelope(0, SIGMA_AOM, t_pulse, Omega_peak=1.0)
    rabi_env      = eval_envelope(coeff, t)   # sqrt(intensity), range [0, 1]
    intensity_model = rabi_env**2             # intensity, range [0, 1]

    # --- Plot ---
    ax.plot(t * 1e6, sig_norm,        color='C0', lw=1.2,        label='Data (intensity)')
    ax.plot(t * 1e6, sig_norm_rabi,        color='C1', lw=1.2,        label='Data (Rabi)')
    ax.plot(t * 1e6, intensity_model, color='C0', lw=1.0, ls='--', label='Model (intensity)')
    ax.plot(t * 1e6, rabi_env,        color='C1', lw=1.0, ls='--',  label='Model (Rabi freq)')



    ax.set_title(f'{t_pulse*1e9:.0f} ns', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.15, 0.23)
    # ax.set_xlim(-0.3, 0.1) # check turn on time
    # ax.set_xlim(t_pulse*1e6 - 0.1, t_pulse*1e6 + 0.1) # check turn off time


# Shared axis labels and legend
for ax in axes[-1]:
    ax.set_xlabel('Time [µs]', fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel('Norm. intensity / Rabi freq', fontsize=7)

axes_flat[0].legend(fontsize=6, loc='upper right')
fig.suptitle(
    f'AOM model validation — erf_rabi_envelope(σ= (90ns on / 58.5 ns off),\n T_on = 0, t_off = t_pulse-55ns)\n'
    ,
    fontsize=11)
plt.tight_layout()
plt.show()

# fig.savefig("Figures/aom_pulse_valid_full.png", dpi=300, bbox_inches="tight", facecolor="white")  

