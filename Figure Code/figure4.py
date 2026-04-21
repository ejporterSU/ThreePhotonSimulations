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
scan_labels  = [r'$T = 10\ \mu$s', r'$T = 1000\ \mu$s', r'$T = 3000$ ms']
scan_colors  = ["#167E29", "#5D2BF5", "#FF6B35"]
scan_markers = ['o', 's', '^']
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
    return -A * np.sin(phi + phi_0) + offset


def batman(x, n, P0, C):
    return n / np.sqrt(1 - ((P0 - x) / (C / 2)) ** 2)


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
    t_3_plot = np.logspace(-6, -2, 1000)* 1e6
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
    c_1_popt, _ = curve_fit(contrast_decay, t_1_data, c_1_data, p0=[max(c_1_data), t_1_data[-1]])
    c_1_fit = contrast_decay(t_1_plot, *c_1_popt)
    
    c_3a_popt, _ = curve_fit(contrast_decay, t_3a_data, c_3a_data, p0=[max(c_3a_data), t_3a_data[-1]])
    c_3a_fit = contrast_decay(t_3_plot, *c_3a_popt)

    c_3b_popt, _ = curve_fit(contrast_decay, t_3b_data, c_3b_data, p0=[max(c_3b_data), t_3b_data[-1]])
    c_3b_fit = contrast_decay(t_3_plot, *c_3b_popt)

    c_3se_popt, _ = curve_fit(contrast_decay, t_3se_data, c_3se_data, p0=[max(c_3se_data), t_3se_data[-1]])
    c_3se_fit = contrast_decay(t_3_plot, *c_3se_popt)

    sens_1 = c_1_fit * (t_1_plot * 1e6)
    sens_3a = c_3a_fit * (t_3_plot * 1e6)
    sens_3b = c_3b_fit * (t_3_plot * 1e6)

    # ── Figure layout ──────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(11, 4.5))
    gs = GridSpec(3, 2, width_ratios=[2.5, 1.8], wspace=0.45, hspace=0.03,
                    left=0.07, right=0.96, top=0.93, bottom=0.13)
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p2 = fig.add_subplot(gs[1, 0], sharex=ax_p1)
    ax_p3 = fig.add_subplot(gs[2, 0], sharex=ax_p1)
    ax    = fig.add_subplot(gs[:, 1])

    phase_axes = [ax_p1, ax_p2, ax_p3]

    # ── Phase scan panels ──────────────────────────────────────────────────────


    phi_fit = np.linspace(0, 4*np.pi, 1000)
    for i, (axi, t_scan, lbl, col, mkr) in enumerate(
            zip(phase_axes, scan_times, scan_labels, scan_colors, scan_markers)):
        fpath = f"{_DATA_DIR}/{fname_base_hist}{t_scan}us.json"
        with open(fpath) as f:
            data = json.load(f)

        phase = data["phases"]
        P_data = data["xc_avg"]
        P_std = data["xc_std"]
        popt = data["sine_popt"]
        popt2 = data["batman_popt"]
        p2p = data["p2p"]

        P_fit = fringe(phi_fit, *popt)

        axi.errorbar(phase, P_data, P_std, fmt=mkr, color=col,
                 markersize=4.5, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        axi.plot(phi_fit, P_fit, '--', color='black', linewidth=LW_FIT, zorder=5)


        counts, bin_edges = np.histogram(data["xc"], bins=25, range=(0,1))

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = (bin_edges[-1] - bin_edges[0]) / len(counts)

        hist_ax = axi.inset_axes([1.005, 0, 0.18, 1])
        hist_ax.barh(bin_centers, counts, height=bin_width,
                     color=col, alpha=0.5, edgecolor='k', linewidth=0.5)

        n_p, P0, C = popt2
        y_fit = np.linspace(P0 - C / 2, P0 + C / 2, 500)
        # valid = np.abs(P0 - y_fit) < (C / 2) * 0.995
        x_fit = np.full_like(y_fit, np.nan)
        x_fit = batman(y_fit, n_p, P0, C)
        hist_ax.plot(x_fit, y_fit, color='red', linewidth=0.7)

        hist_ax.set_ylim(-0.1, 1.1)
        hist_ax.set_xlim(0, max(counts) * 1.1)
        hist_ax.set_yticks([])
        hist_ax.set_xticks([])
        hist_ax.tick_params(axis='x', labelsize=FS_TICK)

        hist_ax.spines[['top', 'right', 'bottom']].set_visible(False)

        # y_max = popt2[1] + popt2[2]/2
        # y_min = popt2[1] - popt2[2]/2
        # ## this discrepency between histogram and density is giving offset but can resolve later
        # if i == 0:
        #     # Double-headed arrow spanning y_min → y_max, labeled C, placed within the hlines
        #     x_arr = X_RIGHT + 0.17
        #     axi.annotate('', xy=(x_arr, y_min), xytext=(x_arr, y_max),
        #                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.2),
        #                  zorder=7)
        #     axi.text(X_RIGHT + 0.35, (y_min + y_max) / 2, r'$C$',
        #              ha='left', va='center', fontsize=FS_CORNER + 2, color='black', zorder=7)

        # for y_val in [y_min, y_max]:
        #     axi.hlines(y_val, X_RIGHT, X_RIGHT + 0.3,
        #                color='black', linewidth=1.2, zorder=5)

        # axi.axvline(X_RIGHT, color='#cccccc', linewidth=0.6, zorder=1)


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
        axi.text(0.02, 0.92, lbl, transform=shadow_tf,
                 ha='left', va='top', fontsize=FS_CORNER, color="#161414",
                 bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                           facecolor='#bbbbbb', edgecolor='none', alpha=0.7), zorder=4)
        axi.text(0.02, 0.92, lbl, transform=axi.transAxes,
                 ha='left', va='top', fontsize=FS_CORNER, color='k',
                 bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.05',
                           facecolor='white', edgecolor='k', linewidth=0.6), zorder=5)

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
    add_panel_label(ax, 'b)')

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

    ax.errorbar(t_3se_data, c_3se_data, yerr=c_3se_data_error, fmt='^', color='purple',
                ms=6, mec='white', mew=1, elinewidth=1, capsize=3, zorder=4, label='3-photon (spin echo)')
    ax.plot(t_3_plot, c_3se_fit, color='purple', linewidth=LW_MAIN, zorder=3)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Ramsey delay $T$ ($\mu$s)', fontsize=FS_LABEL)
    ax.set_ylabel('Contrast', fontsize=FS_LABEL, labelpad=2)
    ax.set_xlim(0.7, 4e4)
    ax.set_ylim(4e-3, 2)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['1', '10', '100', r'$10^3$', r'$10^4$'], fontsize=FS_TICK)

    ax.set_yticks([1e-2, 0.1, 1])
    ax.set_yticklabels(['0.01', '0.1', '1'], fontsize=FS_TICK)
    legend_pos = (0.55, 0.02)  # (x, y) in axes coordinates
    ax.legend(frameon=False, fontsize=11,
              handlelength=1, handletextpad=0.2,
              loc='lower center', bbox_to_anchor=legend_pos)

    # ── Twin axis: Sensitivity C·T ─────────────────────────────────────────────
    ax2 = ax.twinx()  
    max_sens1 = max(sens_1)

    ax2.plot(t_1_plot , sens_1/max_sens1, '--', color=COLOR_1V_LIGHT, linewidth=LW_AUX, zorder=2)
    ax2.plot(t_3_plot , sens_3a/max_sens1, '--', color=COLOR_3V_LIGHT, linewidth=LW_AUX, zorder=2)
    ax2.plot(t_3_plot , sens_3b/max_sens1, '--', color='orange', linewidth=LW_AUX, zorder=2)

    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 2e3)
    ax2.set_ylabel(r'Relative Sensitivity  $\propto C \cdot T$  ($\mu$s)',
                   fontsize=FS_LABEL, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_yticks([0.1, 1, 10, 100, 1000])
    ax2.set_yticklabels(['0.1', '1', '10', '100', r'$10^3$'],
                        fontsize=FS_TICK, color='gray')
    

    ax2.axhline(max(np.concatenate([sens_3a, sens_3b]))/max_sens1, xmin=0.6, xmax=0.85, linestyle=':', color='black', lw=2)

    # ax2.text(0.5 + 0.45/2, 0.88, "~1000x Enhancement", fontsize=10, 
    #          transform=ax2.transAxes, ha='center')
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, f'fig4')
    plt.show()
