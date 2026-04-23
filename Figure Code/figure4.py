#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import curve_fit

import json
import sys
from pathlib import Path
direc = Path.cwd().resolve()
while direc.name != "ThreePhotonSimulations":
    direc = direc.parent
    
_DATA_DIR = direc / "Data"
_FIGURE_DIR = direc / "Figure Code"
sys.path.insert(0, str(_DATA_DIR))
sys.path.insert(0, str(_FIGURE_DIR))
from fig_style import *




# Phase scan config
scan_times   = [10, 1000, 3000]
scan_labels  = [r'$\tau = 10\ \mu$s', r'$\tau = 1$ ms', r'$\tau = 3$ ms']
# scan_colors  = ["#167E29", "#5D2BF5", "#FF6B35"]
scan_colors  = [COLOR_3V]*3
# scan_markers = ['o', 's', '^']
scan_markers = ['s']*3
# fname_base1 = "1v_phase_contrast_040726_"
# fname_base3_sim = "3v_sim_phase_contrast_040726_"
# fname_base3_seq = "3v_seq_phase_contrast_040726_"
fname_base1 = "ramsey_contrast_1.txt"
fname_base3a = "ramsey_contrast_3a.txt"
fname_base3b = "ramsey_contrast_3b.txt"
fname_base_hist = "phase_contrast_040726_"
fname_base3se = "ramsey_contrast_3se.txt"



def contrast_decay(t, c0, sigma):
    return c0 * np.exp(-(t / sigma) ** 2)


def fringe(phi, A, phi_0, offset):
    return 1 - (A * np.sin(phi + phi_0) + offset)



def make_figure():
    # ── Contrast decay data ────────────────────────────────────────────────────

    # one photon
    t_1_plot = np.logspace(-6, -5, 1000) * 1e6
    fpath1 = f"{_DATA_DIR}/{fname_base1}"
    with open(fpath1) as f:
        data = np.loadtxt(fpath1, delimiter=',', skiprows=2)
    t_1_data = data[:, 0]
    c_1_data = data[:, 1]
    c_1_data_error = data[:, 2]

    # 3 photon
    t_3_plot = np.logspace(-6, -1, 1000)* 1e6
    #sequential
    fpath3a = f"{_DATA_DIR}/{fname_base3a}"
    with open(fpath3a) as f:
        data = np.loadtxt(fpath3a, delimiter=',', skiprows=2)
    t_3a_data = data[:, 0]
    c_3a_data = data[:, 1]
    c_4a_data_error = data[:, 2]

    # simultaneous
    fpath3b = f"{_DATA_DIR}/{fname_base3b}"
    with open(fpath3b) as f:
        data = np.loadtxt(fpath3b, delimiter=',', skiprows=2)
    t_3b_data = data[:, 0]
    c_3b_data = data[:, 1]
    c_3b_data_error = data[:, 2]

    # spin echo
    fpath3se = f"{_DATA_DIR}/{fname_base3se}"
    with open(fpath3se) as f:
        data = np.loadtxt(fpath3se, delimiter=',', skiprows=2)
    t_3se_data = data[:, 0]
    c_3se_data = data[:, 1]
    c_3se_data_error = data[:, 2]
    
 

    # fit contrast decay
    c_1_popt,   c_1_pcov   = curve_fit(contrast_decay, t_1_data,  c_1_data,  p0=[max(c_1_data),  t_1_data[-1]])
    c_1_fit  = contrast_decay(t_1_plot, *c_1_popt)

    c_3a_popt,  c_3a_pcov  = curve_fit(contrast_decay, t_3a_data, c_3a_data, p0=[max(c_3a_data), t_3a_data[-1]])
    c_3a_fit = contrast_decay(t_3_plot, *c_3a_popt)

    c_3b_popt,  c_3b_pcov  = curve_fit(contrast_decay, t_3b_data, c_3b_data, p0=[max(c_3b_data), t_3b_data[-1]])
    c_3b_fit = contrast_decay(t_3_plot, *c_3b_popt)

    c_3se_popt, c_3se_pcov = curve_fit(contrast_decay, t_3se_data, c_3se_data, p0=[max(c_3se_data), t_3se_data[-1]])
    c_3se_fit = contrast_decay(t_3_plot, *c_3se_popt)


    sens_1 = c_1_fit * (t_1_plot * 1e6)
    sens_3a = c_3a_fit * (t_3_plot * 1e6)
    sens_3b = c_3b_fit * (t_3_plot * 1e6)

    # ── Free-fall contrast limit ───────────────────────────────────────────────
    g  = 9.80665          # m/s²
    w0 = 500e-6           # beam waist (m) — tune as needed
    t_ff_s = t_3_plot * 1e-6                          # µs → s
    c_ff = 1 - np.cos(np.pi / 2 * np.exp(- (g * t_ff_s**2 / w0)**2))

    # ── Figure layout ──────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(11, 4.5))
    gs = GridSpec(3, 2, width_ratios=[1.3, 1], wspace=0.25, hspace=0.03,
                    left=0.07, right=0.96, top=0.93, bottom=0.13)
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p2 = fig.add_subplot(gs[1, 0], sharex=ax_p1)
    ax_p3 = fig.add_subplot(gs[2, 0], sharex=ax_p1)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], hspace=0.05)
    ax    = fig.add_subplot(gs_right[0])
    ax_sens = fig.add_subplot(gs_right[1])

    phase_axes = [ax_p1, ax_p2, ax_p3]

    # ── Phase scan panels ──────────────────────────────────────────────────────


    phi_fit = np.linspace(0, 4*np.pi, 1000)
    for i, (axi, t_scan, lbl, col, mkr) in enumerate(
            zip(phase_axes, scan_times, scan_labels, scan_colors, scan_markers)):
        fpath = f"{_DATA_DIR}/{fname_base_hist}{t_scan}us.json"
        with open(fpath) as f:
            data = json.load(f)

        phase = np.array(data["phases"])
        P_data = np.array(data["xc_avg"])
        P_std = np.array(data["xc_std"])
        popt = data["sine_popt"]

        P_fit = fringe(phi_fit, *popt)

        axi.errorbar(phase, P_data, P_std, fmt=mkr, color=col,
                 markersize=6, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        axi.plot(phi_fit, P_fit, '--', color='black', linewidth=LW_FIT, zorder=5)


        axi.set_xlim(-0.5, 4 * np.pi + 0.5)
        axi.set_ylim(-0.1, 1.1)
        axi.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        axi.set_yticks([0, 0.5, 1])
        axi.set_yticklabels(['0', '0.5', '1'], fontsize=FS_TICK)
        axi.tick_params(axis='both', labelsize=FS_TICK)

        if i == 1:
            axi.set_ylabel(r'$^1S_0$ Population', fontsize=FS_LABEL)

        # Time label with shadow
        shadow_tf = axi.transAxes + ScaledTranslation(1.2 / 72, -1.2 / 72, fig.dpi_scale_trans)
        axi.text(0.02, 0.95, lbl, transform=shadow_tf,
                 ha='left', va='top', fontsize=12, color="#161414",
                 bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                           facecolor='none', edgecolor='none', alpha=0.7), zorder=4)
        # axi.text(0.02, 0.92, lbl, transform=axi.transAxes,
        #          ha='left', va='top', fontsize=FS_CORNER, color='k',
        #          bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
        #                    facecolor='white', edgecolor='k', linewidth=0.6), zorder=5)

        if i < 2:
            plt.setp(axi.get_xticklabels(), visible=False)
            axi.tick_params(axis='x', which='both', length=0)
        else:
            axi.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'],
                                fontsize=FS_TICK)
            axi.set_xlabel('Phase (rad)', fontsize=FS_LABEL)

    # ── Panel corner labels ────────────────────────────────────────────────────
    lx = -0.18
    add_panel_label(ax_p1, 'a)', x=lx)
    add_panel_label(ax, 'b)', x=-0.22)
    add_panel_label(ax_sens, 'c)', x=-0.22)

    # ── Right panel: Contrast decay ────────────────────────────────────────────
    
    ax.errorbar(t_1_data, c_1_data, yerr=c_1_data_error, fmt='o', color=COLOR_1V,
                ms=6, mec='white', mew=1, elinewidth=1, capsize=3, zorder=4, label='1-photon')
    ax.plot(t_1_plot , c_1_fit, color=COLOR_1V, linewidth=LW_MAIN, zorder=3)

    ax.errorbar(t_3a_data, c_3a_data, yerr=c_4a_data_error, fmt='s', color=COLOR_3V,
                ms=6, mec='white', mew=1, elinewidth=1, capsize=3, zorder=4, label='3-photon (seq.)')
    ax.plot(t_3_plot, c_3a_fit, color=COLOR_3V, linewidth=LW_MAIN, zorder=3)

    ax.errorbar(t_3b_data, c_3b_data, yerr=c_3b_data_error, fmt='s', color='orange',
                ms=6, mec='white', mew=1, elinewidth=1, capsize=3, zorder=4, label='3-photon (sim.)')
    ax.plot(t_3_plot , c_3b_fit, color='orange', linewidth=LW_MAIN, zorder=3)

    # ax.errorbar(t_3se_data, c_3se_data, yerr=c_3se_data_error, fmt='^', color='purple',
    #             ms=6, mec='white', mew=1, elinewidth=1, capsize=3, zorder=4, label='3-photon (spin echo)')
    # ax.plot(t_3_plot, c_3se_fit, color='purple', linewidth=LW_MAIN, zorder=3)
    ax.plot(t_3_plot, c_ff, ':', color='gray', linewidth=LW_AUX, zorder=2,
                label='free-fall limit')
    


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Contrast', fontsize=FS_LABEL, labelpad=2)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlim(0.7, 2e4)
    ax.set_ylim(3e-2, 1.5)
    
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['1', '10', '100', r'$10^3$', r'$10^4$'], fontsize=FS_TICK)

    ax.set_yticks([0.1, 1])
    ax.set_yticklabels(['0.1', '1'], fontsize=FS_TICK)
    legend_pos = (0.55, 0.02)  # (x, y) in axes coordinates
    ax.legend(frameon=False, fontsize=11,
              handlelength=1, handletextpad=0.2,
              loc='lower center', bbox_to_anchor=legend_pos)

    # ── Bottom right panel: Sensitivity C·T ───────────────────────────────────
    max_sens1 = max(sens_1)

    ax_sens.plot(t_1_plot , sens_1/max_sens1, color=COLOR_1V, linewidth=LW_MAIN, zorder=5, label='1-photon')
    ax_sens.plot(t_3_plot , sens_3a/max_sens1, color=COLOR_3V, linewidth=LW_MAIN, zorder=3, label='3-photon (seq.)')
    ax_sens.plot(t_3_plot , sens_3b/max_sens1, color='orange', linewidth=LW_MAIN, zorder=3, label='3-photon (sim.)')

    ax_sens.set_xscale('log')
    ax_sens.set_yscale('log')
    ax_sens.set_xlim(0.7, 2e4)
    ax_sens.set_ylim(0.1, 2e3)
    ax_sens.set_xlabel(r'Ramsey delay $\tau$x (\mathrm{\mu}$s)', fontsize=FS_LABEL)
    ax_sens.set_ylabel('Rel. Sensitivity', fontsize=FS_LABEL)
    ax_sens.set_xticks([1, 10, 100, 1000, 10000])
    ax_sens.set_xticklabels(['1', '10', '100', r'$10^3$', r'$10^4$'], fontsize=FS_TICK)
    ax_sens.set_yticks([0.1, 1, 10, 100, 1000])
    ax_sens.set_yticklabels(['0.1', '1', '10', '100', r'$10^3$'], fontsize=FS_TICK)

    peak_val = max(np.concatenate([sens_3a, sens_3b])) / max_sens1
    ax_sens.axhline(peak_val, xmin=0.6, xmax=0.85, linestyle=':', color='black', lw=2, zorder=5)
    ax_sens.text(0.55, peak_val, r'$\sim\!900\!\times$ Enhancement',
                 transform=ax_sens.get_yaxis_transform(),
                 va='center', ha='right', fontsize=FS_TICK)
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, f'fig4')
    plt.show()
