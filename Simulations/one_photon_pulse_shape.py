import numpy as np
import matplotlib.pyplot as plt
from simulation_functions import *
#%%
# ── Fixed geometry (not swept) ───────────────────────────────────────────── #
_theta_0,  _theta_0z = np.radians(59.4384), 0.0
_theta_1,  _theta_1z = np.radians(-59.64),  0.0
_quant_axis           = np.array([-1.0, 0.0, 0.0])
_pol_vecs  = (np.array([0.0, 0.0, 1.0]),   # z_hat  (689 nm)
              np.array([0.0, 0.0, 1.0]),   # placeholder 688 nm
              np.array([-1.0, 0.0, 0.0]))  # placeholder 679 nm
_mJ_targets = (+1, 0, 0)

# ── Base parameters ──────────────────────────────────────────────────────── #
BASE = dict(
    # --- laser / AOM ---
    P_689        = (0.229 * 85 + 0.5 ) * 1e-3,          # 689 nm peak power [W]
    power_scale  = 1.0,            # multiplicative fudge factor on P_689
    detuning     = 2*PI * 1e6,     # 689 detuning from target Zeeman sublevel [rad/s]
    sigma_aom    = 20e-9,          # AOM erf rise/fall width [s]
    t0_aom       = 0.0,            # AOM 50% rise point [s]
    # --- atomic cloud ---
    cloud_radii  = [0e-6, 0e-6, 0e-6],  # 1-sigma position widths [x,y,z] [m]
    temperatures = [0e-6,  0e-6,  0e-6],    # thermal widths [x,y,z] [K]
    # --- 689 beam waist ---
    w0 = 0.5e-3,                  # 689 nm 1/e^2 radius [m]
    # --- readout / experiment ---
    B_field_G    = 20.0,           # bias field [G]
    t_push       = 0.0e-6,         # 461 nm push duration for readout [s]
    # --- simulation ---
    T_MAX   = 2e-6,             # max pulse duration scanned [s]
    dt      = 5e-9,               # time step [s]
    N_atoms = 50,                  # atoms per shot
)

# ── Sweep configuration ───────────────────────────────────────────────────── #
# Set ACTIVE_SWEEP to one of the keys below, then run the script.
# Each entry: (list_of_values, label_format_function)

SWEEPS = {
    'sigma_aom':   ([30e-9, 60e-9, 90e-9, 150e-9, 250e-9],
                    lambda v: rf'$\sigma$ = {v*1e9:.0f} ns'),

    'power_scale': ([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    lambda v: f'power × {v:.2f}'),

    'temperature': ([0e-6, 0.5e-6, 2e-6, 5e-6, 10e-6],
                    lambda v: f'T = {v*1e6:.1f} µK'),

    'cloud_radii': ([10e-6, 25e-6, 40e-6, 80e-6, 150e-6],
                    lambda v: f'r_xy = {v*1e6:.0f} µm'),

    'detuning':    ([0.0, 2*PI*0.5e6, 2*PI*1e6, 2*PI*2e6, 2*PI*5e6],
                    lambda v: rf'$\Delta$ = {v/(2*PI)*1e-3:.0f} kHz'),

    'w0':          ([0.3e-3, 0.45e-3, 0.55e-3, 0.7e-3, 1.0e-3],
                    lambda v: f'w₀ = {v*1e3:.2f} mm'),
}

ACTIVE_SWEEP = 'power_scale'   # ← change this to explore a different parameter

# ── Simulation helper ─────────────────────────────────────────────────────── #
def run_sim(p, envelope='ERF'):
    """
    Build k-vectors, sample an ensemble, run simulate_one_photon_rabi_dynamics,
    apply readout, and return (tlist [s], measured_population).

    Args:
        p        (dict):  Parameter dict with the same keys as BASE.
        envelope (str):   'ERF', 'SQUARE', 'GAUSSIAN', or 'BLACKMAN'.
    """
    k_vec_0 = (2*PI / lambda_689) * get_k_hat(_theta_0, _theta_0z)
    k_vec_1 = (2*PI / lambda_688) * get_k_hat(_theta_1, _theta_1z)
    k_vec_2 = (2*PI / lambda_679) * get_k_hat(0.0, 0.0)
    k_vecs  = (k_vec_0, k_vec_1, k_vec_2)

    powers  = (p['P_689'] * p['power_scale'], 0.0, 0.0)
    radii   = (p['w0'], 0.9e-3, 0.9e-3)
    det     = np.array([p['detuning'], 0.0, 0.0])
    ep      = {'t0': p['t0_aom'], 'sigma': p['sigma_aom']}

    pos, vel = sample_atomic_ensemble(
        p['cloud_radii'], p['temperatures'], n_samples=p['N_atoms'])
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)

    tlist, pop = simulate_one_photon_rabi_dynamics(
        pos, vel, radii, powers, det, k_vecs,
        _pol_vecs, _quant_axis, _mJ_targets,
        t_max=p['T_MAX'], dt=p['dt'],
        envelope=envelope,
        envelope_params=ep,
    )
    return tlist, apply_readout(pop, p['t_push'])


# ── Theory reference (base parameters, no ensemble averaging) ─────────────── #
def theory_curve(p, tlist):
    """Analytical damped-Rabi curve for a point atom at beam center."""
    C0           = get_coupling_factor(_pol_vecs[0], _quant_axis, _mJ_targets[0])
    Omega_theory = C0 * gamma_689 * np.sqrt(
        p['P_689'] * p['power_scale'] * 100 / (PI * p['w0']**2) / Is_689)
    Omega_eff    = np.sqrt(Omega_theory**2 + p['detuning']**2)
    decay        = np.exp(-3/4 * tlist * gamma_689)
    pop          = (Omega_theory**2 / Omega_eff**2) * (
                    0.5 - 0.5 * np.cos(Omega_eff * tlist) * decay)
    return apply_readout(pop, p['t_push'])


# ── Main ─────────────────────────────────────────────────────────────────── #
if __name__ == '__main__':
    values, label_fn = SWEEPS[ACTIVE_SWEEP]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis', len(values))

    for idx, val in enumerate(values):
        p = BASE.copy()
        p['cloud_radii']  = list(BASE['cloud_radii'])   # avoid mutating base lists
        p['temperatures'] = list(BASE['temperatures'])

        # Apply sweep value
        if ACTIVE_SWEEP == 'temperature':
            p['temperatures'] = [val, val, val]
        elif ACTIVE_SWEEP == 'cloud_radii':
            # scale xy; keep z proportional to the base xy/z ratio
            p['cloud_radii'] = [val, val, val]
        else:
            p[ACTIVE_SWEEP] = val

        tlist, pop = run_sim(p, envelope='SQUARE')
        ax.plot(tlist * 1e6, pop, color=cmap(idx), label=label_fn(val))

    # Reference: theory for base parameters
    tlist_th = np.linspace(0, BASE['T_MAX'], int(BASE['T_MAX'] / BASE['dt']) + 1)
    ax.plot(tlist_th * 1e6, theory_curve(BASE, tlist_th),
            'k--', alpha=0.6, label='Theory (base)')

    ax.set_xlabel('Pulse duration [µs]')
    ax.set_ylabel('Excited state population (after readout)')
    ax.set_title(f'689 nm Rabi — sweep: {ACTIVE_SWEEP}')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
