#%%
"""
three_photon_speedup_testing.py
--------------------------------
Runs the three-photon simulation with both the original QuTiP-based
function and the new numpy/scipy function (Strategies A+B+C from
planning.md), using IDENTICAL parameters to three_photon_validate.py.

Reports wall-clock time for each method and plots results overlaid so
you can verify they agree.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from simulation_functions import (
    simulate_three_photon_rabi_dynamics,
    simulate_three_photon_rabi_dynamics_new,
    sample_atomic_ensemble,
    get_zeeman_detuning,
    get_k_hat,
    G_J_3P1, lambda_689, lambda_688, lambda_679, PI,
)

# ── Identical setup to three_photon_validate.py ──────────────────────── #

# Beam propagation angles
theta_0,  theta_0z = np.radians(59.4384), 0.0
theta_1,  theta_1z = np.radians(-59.64),  0.0
theta_2,  theta_2z = 0.0,                 0.0

k_vec_0 = (2*PI / lambda_689) * get_k_hat(theta_0, theta_0z)
k_vec_1 = (2*PI / lambda_688) * get_k_hat(theta_1, theta_1z)
k_vec_2 = (2*PI / lambda_679) * get_k_hat(theta_2, theta_2z)
k_vecs  = (k_vec_0, k_vec_1, k_vec_2)

quant_axis = np.array([-1.0, 0.0, 0.0])

pol_vecs = (
    np.array([0.0, 0.0,  1.0]),
    np.array([0.0, 0.0,  1.0]),
    np.array([-1.0, 0.0, 0.0]),
)
mJ_targets = (+1, -1, 0)

# Laser powers
pd_mv_689 = 85
pd_mv_679 = 33
pd_mv_688 = 80
P_689 = 1e-3 * (0.229 * pd_mv_689 - 0.586)
P_679 = 1e-3 * (0.146 * pd_mv_679 + 0.008)
P_688 = 1e-3 * (0.133 * pd_mv_688 + 0.471)
powers = [P_689, P_688, P_679]

# B-field and detunings
B_field_G = 20
B_field_T = B_field_G * 1e-4
delta_zeeman_689 = get_zeeman_detuning(G_J_3P1, mJ_targets[0], B_field_T)
detuning_mj1 = 2*PI * 5.0e6
detuning_0   = delta_zeeman_689 + detuning_mj1
detuning_2   = 2*PI * -400e6
detunings_base = [detuning_mj1,
                  detuning_2 - detuning_mj1,
                  detuning_2]

# AC Stark correction (applied in TIME mode in validate script)
delta_AC = 2 * PI * 1e6 * 0.982
detunings = list(detunings_base)
detunings[2] = detunings[2] + delta_AC

# AOM / beam parameters
sigma_aom  = 90e-9
ep         = {'t0': 0.0, 'sigma': sigma_aom}
envelope   = 'ERF'
cloud_radii  = [0e-6, 0e-6, 0e-6]
temperatures = [0e-6, 0e-6, 0e-6]
w0_689       = 0.54e-3
w0_688       = 0.9e-3
w0_679       = 0.9e-3
beam_radii   = [w0_689, w0_688, w0_679]

# Simulation parameters
T_MAX   = 10e-6
dt      = 50e-9
N_atoms = 50
n_shots = 50

# Sample ensemble (same seed for both runs)
rng = np.random.default_rng(42)
pos, vel = sample_atomic_ensemble(cloud_radii, temperatures, n_samples=N_atoms)
pos = np.atleast_2d(pos)
vel = np.atleast_2d(vel)

print("=" * 60)
print("Parameters (matching three_photon_validate.py, TIME mode)")
print(f"  Powers   689={P_689*1e3:.2f} mW  688={P_688*1e3:.2f} mW  679={P_679*1e3:.2f} mW")
print(f"  Detunings  d0={detunings[0]/(2*PI)*1e-6:.2f} MHz  "
      f"d1={detunings[1]/(2*PI)*1e-6:.2f} MHz  "
      f"d2={detunings[2]/(2*PI)*1e-6:.2f} MHz")
print(f"  T_MAX={T_MAX*1e6:.1f} µs  dt={dt*1e9:.0f} ns  "
      f"N_atoms={N_atoms}  n_shots={n_shots}")
print("=" * 60)

# ── Run OLD simulation (QuTiP) ────────────────────────────────────────── #
# NOTE: The old function's SQUARE mode has a numpy casting bug in newer numpy
# (QuTiP returns complex arrays; float64 avg_populations can't absorb them).
# This test uses ERF mode only — which is also what three_photon_validate.py uses.
print("\n[1/3] Running OLD method (QuTiP mesolve)...")
t_start_old = time.perf_counter()

tlist_old, pops_old = simulate_three_photon_rabi_dynamics(
    pos, vel, beam_radii, powers, list(detunings), k_vecs,
    pol_vecs, quant_axis, mJ_targets,
    t_max=T_MAX, dt=dt,
    n_shots=n_shots,
    envelope=envelope,
    envelope_params=ep,
)

t_old = time.perf_counter() - t_start_old
print(f"  → Done in {t_old:.2f} s")

# ── Run NEW simulation (numpy, sequential) ────────────────────────────── #
print("\n[2/3] Running NEW method (numpy/scipy, n_jobs=1, Strategies A+B)...")
t_start_new1 = time.perf_counter()

tlist_new1, pops_new1 = simulate_three_photon_rabi_dynamics_new(
    pos, vel, beam_radii, powers, list(detunings), k_vecs,
    pol_vecs, quant_axis, mJ_targets,
    t_max=T_MAX, dt=dt,
    n_shots=n_shots,
    envelope=envelope,
    envelope_params=ep,
    n_jobs=1,
)

t_new1 = time.perf_counter() - t_start_new1
print(f"  → Done in {t_new1:.2f} s")

# ── Run NEW simulation (numpy, parallel) ─────────────────────────────── #
print("\n[3/3] Running NEW method (numpy/scipy, n_jobs=-1, Strategies A+B+C)...")
t_start_new_par = time.perf_counter()

tlist_new_par, pops_new_par = simulate_three_photon_rabi_dynamics_new(
    pos, vel, beam_radii, powers, list(detunings), k_vecs,
    pol_vecs, quant_axis, mJ_targets,
    t_max=T_MAX, dt=dt,
    n_shots=n_shots,
    envelope=envelope,
    envelope_params=ep,
    n_jobs=-1,
)

t_new_par = time.perf_counter() - t_start_new_par
print(f"  → Done in {t_new_par:.2f} s")

# ── Timing summary ────────────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print(f"  Old  (QuTiP,  sequential)  : {t_old:.2f} s")
print(f"  New  (numpy,  sequential)  : {t_new1:.2f} s   "
      f"[{t_old/t_new1:.1f}x speedup — Strategies A+B]")
print(f"  New  (numpy,  parallel  )  : {t_new_par:.2f} s   "
      f"[{t_old/t_new_par:.1f}x speedup — Strategies A+B+C]")
print("=" * 60)

# ── Numerical agreement check ─────────────────────────────────────────── #
# Interpolate old onto new time grid (they share tlist since same n_shots)
max_abs_diff_seq = np.max(np.abs(pops_old - pops_new1))
max_abs_diff_par = np.max(np.abs(pops_old - pops_new_par))
print(f"\nMax |old − new_sequential| across all states & times : {max_abs_diff_seq:.2e}")
print(f"Max |old − new_parallel  | across all states & times : {max_abs_diff_par:.2e}")
if max_abs_diff_seq < 1e-3:
    print("  ✓ Sequential new method agrees with old (< 1e-3)")
else:
    print("  ✗ Sequential new method DIFFERS from old — check tolerances")

# ── Plot comparison ───────────────────────────────────────────────────── #
state_labels = ['1S0', '3P1', '3P0', '3P2']
colors       = ['C0', 'C1', 'C3', 'C4']
t_us_old     = tlist_old     * 1e6
t_us_new1    = tlist_new1    * 1e6
t_us_new_par = tlist_new_par * 1e6

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: old vs new_sequential
ax = axes[0]
for k in range(4):
    ax.scatter(t_us_old,  pops_old[k],  color=colors[k], s=40,
               label=f'{state_labels[k]} old',  alpha=0.9, zorder=3)
    ax.plot(t_us_new1, pops_new1[k], color=colors[k], lw=1.5,
            ls='--', label=f'{state_labels[k]} new (A+B)')
ax.set_xlabel('Time [µs]')
ax.set_ylabel('State population')
ax.set_title(f'Old (QuTiP, {t_old:.1f}s) vs New sequential (numpy, {t_new1:.1f}s)\n'
             f'Speedup A+B: {t_old/t_new1:.1f}×   |max diff| = {max_abs_diff_seq:.1e}')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7, ncol=2)

# Right panel: old vs new_parallel
ax = axes[1]
for k in range(4):
    ax.scatter(t_us_old,     pops_old[k],     color=colors[k], s=40,
               label=f'{state_labels[k]} old',      alpha=0.9, zorder=3)
    ax.plot(t_us_new_par, pops_new_par[k], color=colors[k], lw=1.5,
            ls='--', label=f'{state_labels[k]} new (A+B+C)')
ax.set_xlabel('Time [µs]')
ax.set_ylabel('State population')
ax.set_title(f'Old (QuTiP, {t_old:.1f}s) vs New parallel (numpy, {t_new_par:.1f}s)\n'
             f'Speedup A+B+C: {t_old/t_new_par:.1f}×   |max diff| = {max_abs_diff_par:.1e}')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7, ncol=2)

plt.suptitle('Three-photon simulation: QuTiP vs numpy/scipy (ERF mode)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved to speedup_comparison.png")
