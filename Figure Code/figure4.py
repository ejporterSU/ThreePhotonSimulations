#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
})

# =============================================================
# Configuration
# =============================================================

VERTICAL = True

# Font sizes
FS_LABEL  = 15
FS_TICK   = 12
FS_CORNER = 9
FS_RLABEL = 15
FS_RTICK  = 12

# Line widths
LW_MAIN = 2.5
LW_SENS = 2.0
LW_FIT  = 1.5

# Colors
c1  = "#1845D4"
c3  = "#C41E1E"
c1s = "#90AAFF"
c3s = "#FF9090"

# Phase scan config
scan_times   = [10e-6, 500e-6, 2e-3]
scan_labels  = [r'$T = 10\ \mu$s', r'$T = 500\ \mu$s', r'$T = 2$ ms']
scan_colors  = ["#167E29", "#5D2BF5", "#FF6B35"]
scan_markers = ['o', 's', '^']

# =============================================================
# Physics models
# =============================================================

tau_1 = 3.5e-6
tau_3 = 2e-3
c0_1  = 0.82
c0_3  = 0.65


def contrast_decay(t, c0, tau):
    return c0 * np.exp(-(t / tau) ** 2)


def fringe(phi, A, phi_0, offset):
    return offset + A * np.cos(phi + phi_0)


# =============================================================
# Generate contrast decay data
# =============================================================

t_1 = np.logspace(-6, -4.5, 1000)
t_3 = np.logspace(-6, -1.8, 1000)

np.random.seed(42)
t_1_data = np.logspace(-6, -5, 10)
t_3_data = np.logspace(-4, -2.5, 10)

c_1 = contrast_decay(t_1, c0_1, tau_1)
c_3 = contrast_decay(t_3, c0_3, tau_3)
c_1_data = contrast_decay(t_1_data, c0_1, tau_1) + np.random.normal(0, 0.005, len(t_1_data))
c_3_data = contrast_decay(t_3_data, c0_3, tau_3) + np.random.normal(0, 0.01, len(t_3_data))

sens_1 = c_1 * (t_1 * 1e6)
sens_3 = c_3 * (t_3 * 1e6)

# =============================================================
# Figure layout
# =============================================================

if VERTICAL:
    fig = plt.figure(figsize=(6.5, 9))
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_outer = GridSpec(2, 1, height_ratios=[3, 2.2], hspace=0.35,
                        left=0.13, right=0.87, top=0.95, bottom=0.08)
    gs_inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0], hspace=0.0)
    ax_p1 = fig.add_subplot(gs_inner[0])
    ax_p2 = fig.add_subplot(gs_inner[1], sharex=ax_p1)
    ax_p3 = fig.add_subplot(gs_inner[2], sharex=ax_p1)
    ax    = fig.add_subplot(gs_outer[1])
else:
    fig = plt.figure(figsize=(11, 4.5))
    gs = GridSpec(3, 2, width_ratios=[2.5, 1.8], wspace=0.18, hspace=0.03,
                  left=0.07, right=0.96, top=0.93, bottom=0.13)
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p2 = fig.add_subplot(gs[1, 0], sharex=ax_p1)
    ax_p3 = fig.add_subplot(gs[2, 0], sharex=ax_p1)
    ax    = fig.add_subplot(gs[:, 1])

phase_axes = [ax_p1, ax_p2, ax_p3]

# =============================================================
# Phase scan panels (with KDE marginals on right edge)
# =============================================================

phi_nominal = np.linspace(0, 4 * np.pi, 250)
phi_fit     = np.linspace(0, 4 * np.pi, 500)
phase_noise_std = 0.5
rng = np.random.default_rng(7)

# Extra x-space for the KDE strip
KDE_WIDTH = 1.5   # width in phase-axis units
X_RIGHT   = 4 * np.pi + 0.3                  # where the KDE strip starts
X_LIM     = X_RIGHT + KDE_WIDTH + 0.2        # right edge of axes

for i, (axi, t_scan, lbl, col, mkr) in enumerate(
        zip(phase_axes, scan_times, scan_labels, scan_colors, scan_markers)):

    # --- Generate synthetic fringe data ---
    C = contrast_decay(t_scan, c0_3, tau_3)
    phi_0_true = 0.25 + i * 0.15

    delta_phi = rng.normal(0, phase_noise_std, len(phi_nominal))
    P_data = fringe(phi_nominal + delta_phi, C / 2, phi_0_true, 0.5) \
             + rng.normal(0, 0.015, len(phi_nominal))
    P_data = np.clip(P_data, 0, 1)

    # --- Fit ---
    p0 = [C / 2, phi_0_true, 0.5]
    popt, _ = curve_fit(fringe, phi_nominal, P_data, p0=p0,
                        bounds=([0, -np.pi, 0], [0.6, np.pi, 1]))
    P_fit = fringe(phi_fit, *popt)

    # --- Scatter + fit curve ---
    axi.plot(phi_nominal, P_data, mkr, color=col,
             markersize=4.5, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
    axi.plot(phi_fit, P_fit, '--', color='black', linewidth=2, zorder=5)

    # --- KDE marginal drawn on the main axes ---
    kde = gaussian_kde(P_data, bw_method=0.3)
    y_grid = np.linspace(-0.1, 1.1, 300)
    density = kde(y_grid)
    density_scaled = X_RIGHT + (density / density.max()) * KDE_WIDTH

    axi.fill_betweenx(y_grid, X_RIGHT, density_scaled,
                       color=col, alpha=0.3, zorder=2)
    axi.plot(density_scaled, y_grid,
             color=col, linewidth=1.0, alpha=0.7, zorder=2)

    # Arcsine overlay on the KDE strip
    A_fit, _, offset_fit = popt
    # eps = 1e-4
    # P_arc = np.linspace(offset_fit - A_fit + eps, offset_fit + A_fit - eps, 2000)
    # pdf_arc = 1.0 / (np.pi * np.sqrt(A_fit**2 - (P_arc - offset_fit)**2))
    # pdf_arc_scaled = X_RIGHT + (pdf_arc / pdf_arc.max()) * KDE_WIDTH
    # mask = pdf_arc / pdf_arc.max() <= 1.0
    # axi.plot(pdf_arc_scaled[mask], P_arc[mask],
    #          color=c3, linewidth=LW_FIT, zorder=4)
    y_max = offset_fit + A_fit
    y_min = offset_fit - A_fit

    for y_val in [y_min, y_max]:
        axi.hlines(y_val, X_RIGHT, X_RIGHT + 0.3,
                color='black', linewidth=1.2, zorder=5)

    # Thin separator line between data region and KDE strip
    axi.axvline(X_RIGHT, color='#cccccc', linewidth=0.6, zorder=1)

    # --- Panel formatting ---
    axi.set_xlim(-0.5, X_LIM)
    axi.set_ylim(-0.1, 1.1)
    axi.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    axi.set_yticks([0, 0.5, 1])
    axi.set_yticklabels(['0', '0.5', '1'], fontsize=FS_TICK)
    axi.tick_params(axis='both', labelsize=FS_TICK)

    if i == 1:
        axi.set_ylabel(r'Population ($^3P_0$)', fontsize=FS_LABEL)

    # Time label with shadow
    shadow_tf = axi.transAxes + ScaledTranslation(1.2 / 72, -1.2 / 72, fig.dpi_scale_trans)
    axi.text(0.02, 0.92, lbl, transform=shadow_tf,
             ha='left', va='top', fontsize=FS_CORNER, color="#161414",
             bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                       facecolor='#bbbbbb', edgecolor='none', alpha=0.7), zorder=4)
    axi.text(0.02, 0.92, lbl, transform=axi.transAxes,
             ha='left', va='top', fontsize=10, color='k',
             bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                       facecolor='white', edgecolor='k', linewidth=0.6), zorder=5)

    # Only bottom panel gets x-tick labels
    if i < 2:
        plt.setp(axi.get_xticklabels(), visible=False)
        axi.tick_params(axis='x', which='both', length=0)
    else:
        axi.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'],
                            fontsize=FS_TICK)
        axi.set_xlabel('Phase (rad)', fontsize=FS_LABEL)

# =============================================================
# Panel corner labels
# =============================================================

_lx = -0.14 if VERTICAL else -0.18
ax_p1.text(_lx, 1.05, 'a)', transform=ax_p1.transAxes,
           fontsize=12, fontweight='bold', va='top', ha='left')
_lx2 = -0.14
ax.text(_lx2, 1.02, 'b)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left')

# =============================================================
# Right panel: Contrast decay
# =============================================================

ax.plot(t_1 * 1e6, c_1, color=c1, linewidth=LW_MAIN, zorder=3)
ax.plot(t_3 * 1e6, c_3, color=c3, linewidth=LW_MAIN, zorder=3)
ax.plot(t_1_data * 1e6, c_1_data, 'o', color=c1,
        markeredgecolor='white', markeredgewidth=0.8, markersize=7, zorder=4, label='1-photon')
ax.plot(t_3_data * 1e6, c_3_data, 's', color=c3,
        markeredgecolor='white', markeredgewidth=0.8, markersize=7, zorder=4, label='3-photon')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Ramsey delay $T$ ($\mu$s)', fontsize=FS_RLABEL)
ax.set_ylabel('Contrast', fontsize=FS_RLABEL, fontweight='bold', labelpad=2)
ax.set_xlim(0.7, 2e4)
ax.set_ylim(4e-3, 3)
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(['1', '10', '100', r'$10^3$', r'$10^4$'], fontsize=FS_RTICK)
ax.set_yticks([1e-2, 0.1, 1])
ax.set_yticklabels(['0.01', '0.1', '1'], fontsize=FS_RTICK)
ax.legend(frameon=False, fontsize=FS_RTICK, loc='upper left',
          handlelength=1.5, handletextpad=0.5)

# Enhancement arrow
y_arrow = 0.03
ax.annotate('', xy=(4e3, y_arrow), xytext=(5.5, y_arrow),
            arrowprops=dict(arrowstyle='<->, head_width=0.5, head_length=1',
                            color='black', lw=3),
            zorder=6)
ax.text(130, y_arrow * 1.1, r'$\sim$1000$\times$ Enhancement',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=6)

# =============================================================
# Twin axis: Sensitivity C·T
# =============================================================

ax2 = ax.twinx()
ax2.plot(t_1 * 1e6, sens_1, '--', color=c1s, linewidth=LW_SENS, zorder=2)
ax2.plot(t_3 * 1e6, sens_3, '--', color=c3s, linewidth=LW_SENS, zorder=2)
ax2.set_yscale('log')
ax2.set_ylim(0.1, 2e3)
ax2.set_ylabel(r'Sensitivity  $\propto C \cdot T$  ($\mu$s)',
               fontsize=FS_RLABEL, fontweight='bold', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_xlim(ax.get_xlim())
ax2.set_yticks([0.1, 1, 10, 100, 1000])
ax2.set_yticklabels(['0.1', '1', '10', '100', r'$10^3$'],
                    fontsize=FS_RTICK, color='gray')

# =============================================================
# Save
# =============================================================

plt.show()

suffix = "v" if VERTICAL else "h"
fig.savefig(f"Figures/fig4{suffix}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"Figures/fig4{suffix}.png", dpi=150, bbox_inches="tight")
# %%
