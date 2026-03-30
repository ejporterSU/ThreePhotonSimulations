#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# --- Layout toggle ---
VERTICAL = True   # True  → phase scans stacked above contrast plot
                   # False → phase scans left, contrast right (default)

# --- Style constants ---
FS_LABEL  = 10      # axis labels (left panels)
FS_TICK   = 9       # tick labels (left panels)
FS_CORNER = 9       # in-panel time labels
FS_RLABEL = 12      # axis labels (right panel)
FS_RTICK  = 10      # tick labels (right panel)
LW_MAIN   = 2.5     # solid theory curves
LW_SENS   = 2.0     # dashed sensitivity curves
LW_FIT    = 1.5     # sine / arcsine fits

# --- Parameters ---
tau_1 = 3.5e-6   # 1-photon coherence time (s)
tau_3 = 2e-3     # 3-photon coherence time (s)
c0_1  = 0.82
c0_3  = 0.65

def contrast_decay(t, c0, tau):
    return c0 * np.exp(-(t / tau)**2)

# --- Time arrays ---
t_1 = np.logspace(-6, -4.5, 1000)
t_3 = np.logspace(-6, -1.8, 1000)

np.random.seed(42)
t_1_data = np.logspace(-6, -5, 10)
t_3_data = np.logspace(-4, -2.5, 10)

c_1 = contrast_decay(t_1, c0_1, tau_1)
c_3 = contrast_decay(t_3, c0_3, tau_3)
c_1_data = contrast_decay(t_1_data, c0_1, tau_1) + np.random.normal(0, 0.005, len(t_1_data))
c_3_data = contrast_decay(t_3_data, c0_3, tau_3) + np.random.normal(0, 0.01,  len(t_3_data))

# Sensitivity figure of merit: C(T) · T
sens_1 = c_1 * (t_1 * 1e6)
sens_3 = c_3 * (t_3 * 1e6)

# --- Colors ---
c1  = "#1845D4"
c3  = "#C41E1E"
c1s = "#90AAFF"
c3s = "#FF9090"

# --- Layout ---
plt.rcParams.update({'font.size': FS_TICK, 'font.family': 'sans-serif'})

if VERTICAL:
    fig = plt.figure(figsize=(6.5, 9))
    # Outer: phase-scan block (top) + contrast plot (bottom), with room for xlabel between them
    gs_outer = GridSpec(2, 1, height_ratios=[3, 2.2], hspace=0.35,
                        left=0.13, right=0.87, top=0.95, bottom=0.08)
    # Inner: three tightly stacked phase panels
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

# Histogram axes glued flush to the right of each phase panel
dividers  = [make_axes_locatable(a) for a in [ax_p1, ax_p2, ax_p3]]
hist_axes = [div.append_axes("right", size="20%", pad=0.05) for div in dividers]

# --- Phase scan data ---
def fringe(phi, A, phi_0, offset):
    return offset + A * np.cos(phi + phi_0)

scan_times   = [10e-6, 500e-6, 2e-3]
scan_labels  = [r'$T = 10\ \mu$s', r'$T = 500\ \mu$s', r'$T = 2$ ms']
scan_colors  = ["#167E29", "#5D2BF5", '#FF6B35']
scan_markers = ['o', 's', '^']
phi_nominal = np.linspace(0, 4 * np.pi, 250)
phi_fit     = np.linspace(0, 4 * np.pi, 500)
phase_noise_std = 0.5   # rad

scan_axes = [ax_p1, ax_p2, ax_p3]
rng = np.random.default_rng(7)

for i, (axi, t_scan, lbl, col, mkr) in enumerate(
        zip(scan_axes, scan_times, scan_labels, scan_colors, scan_markers)):

    C = contrast_decay(t_scan, c0_3, tau_3)
    phi_0_true = 0.25 + i * 0.15

    delta_phi = rng.normal(0, phase_noise_std, len(phi_nominal))
    P_data = fringe(phi_nominal + delta_phi, C / 2, phi_0_true, 0.5) \
             + rng.normal(0, 0.015, len(phi_nominal))
    P_data = np.clip(P_data, 0, 1)

    p0 = [C / 2, phi_0_true, 0.5]
    popt, _ = curve_fit(fringe, phi_nominal, P_data, p0=p0,
                        bounds=([0, -np.pi, 0], [0.6, np.pi, 1]))
    P_fit = fringe(phi_fit, *popt)

    axi.plot(phi_nominal, P_data, mkr, color=col,
             markersize=4.5, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    axi.plot(phi_fit, P_fit, '--', color='black', linewidth=2, zorder=5)

    # Marginal histogram
    ax_h = hist_axes[i]
    ax_h.hist(P_data, bins=15, orientation='horizontal',
              color=col, alpha=0.75, edgecolor='black', density=True, linewidth=0.5)

    # Arcsine distribution, clipped to histogram peak
    A_fit, _, offset_fit = popt
    hist_counts, _ = np.histogram(P_data, bins=25, density=True)
    max_density = hist_counts.max()

    eps = 1e-4
    P_arc = np.linspace(offset_fit - A_fit + eps, offset_fit + A_fit - eps, 2000)
    pdf_arc = 1.0 / (np.pi * np.sqrt(A_fit**2 - (P_arc - offset_fit)**2))
    mask = pdf_arc <= max_density
    ax_h.plot(pdf_arc[mask], P_arc[mask], color=c3, linewidth=LW_FIT, zorder=4)

    ax_h.set_ylim(-0.1, 1.1)
    ax_h.set_yticks([])
    ax_h.set_xticks([])
    for sp in ax_h.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('#bbbbbb')
    if i < 2:
        plt.setp(ax_h.get_xticklabels(), visible=False)
        ax_h.tick_params(axis='x', which='both', length=0)

    axi.set_xlim(-0.5, 4 * np.pi + 0.5)
    axi.set_ylim(-0.1, 1.1)
    axi.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    axi.set_yticks([0, 0.5, 1])
    axi.set_yticklabels(['0', '0.5', '1'], fontsize=FS_TICK)
    axi.tick_params(axis='both', labelsize=FS_TICK)
    if i == 1:
        axi.set_ylabel(r'Population ($^3P_0$)', fontsize=FS_LABEL)

    shadow_tf = axi.transAxes + ScaledTranslation(1.2/72, -1.2/72, fig.dpi_scale_trans)
    axi.text(0.98, 0.92, lbl, transform=shadow_tf,
             ha='right', va='top', fontsize=FS_CORNER, color="#161414",
             bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                       facecolor='#bbbbbb', edgecolor='none', alpha=0.7), zorder=4)
    axi.text(0.98, 0.92, lbl, transform=axi.transAxes,
             ha='right', va='top', fontsize=10, color='k',
             bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                       facecolor='white', edgecolor='k', linewidth=0.6), zorder=5)

    if i < 2:
        plt.setp(axi.get_xticklabels(), visible=False)
        axi.tick_params(axis='x', which='both', length=0)
    else:
        axi.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], fontsize=FS_TICK)
        axi.set_xlabel('Phase (rad)', fontsize=FS_LABEL)

# Panel labels
_lx = -0.14 if VERTICAL else -0.18
ax_p1.text(_lx, 1.05, 'a)', transform=ax_p1.transAxes,
           fontsize=12, fontweight='bold', va='top', ha='left')
_lx2 = -0.14 if VERTICAL else -0.14
ax.text(_lx2, 1.02, 'b)', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left')

# --- Right panel: Contrast decay ---
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

# Arrow: ~1000x coherence time enhancement
y_arrow = 0.03
ax.annotate('', xy=(4e3, y_arrow), xytext=(5.5, y_arrow),
            arrowprops=dict(arrowstyle='<->, head_width=0.5, head_length=1', color='black', lw=3),
            zorder=6)
ax.text(130, y_arrow * 1.1, r'$\sim$1000$\times$ Enhancement',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=6)

# --- Twin y-axis: Sensitivity C·T ---
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
ax2.set_yticklabels(['0.1', '1', '10', '100', r'$10^3$'], fontsize=FS_RTICK, color='gray')

plt.show()


fig.savefig("Figures/fig4v.pdf", dpi=300, bbox_inches="tight", facecolor="white")  # save for figure resolution
fig.savefig("Figures/fig4v.png", dpi=150, bbox_inches="tight")  # save for previewing