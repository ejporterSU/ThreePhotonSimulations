#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import os
import json
_DATA_DIR = "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Data"
_FIGURE_DIR = "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code"
os.chdir(_FIGURE_DIR)
from fig_style import *




# Phase scan config
scan_times   = [10e-6, 1000e-6, 3000e-6]
scan_labels  = [r'$T = 10\ \mu$s', r'$T = 1000\ \mu$s', r'$T = 3000$ ms']
scan_colors  = ["#167E29", "#5D2BF5", "#FF6B35"]
scan_markers = ['o', 's', '^']
fname_base = "phase_contrast_040226_"

# ── Physics models ─────────────────────────────────────────────────────────────
tau_1 = 3.5e-6
tau_3 = 2e-3
c0_1  = 0.82
c0_3  = 0.65



def contrast_decay(t, c0, sigma):
    return c0 * np.exp(-(t / sigma) ** 2)


def fringe(phi, A, phi_0, offset):
    return A * np.sin(phi + phi_0) + offset


def make_figure(vertical=False):
    # ── Contrast decay data ────────────────────────────────────────────────────
    t_1_plot = np.logspace(-6, -5, 1000)
    t_1_data = []
    c_1_data = []

    t_3_plot = np.logspace(-6, -2, 1000)
    t_3_data = np.array([10, 100, 1000, 2000, 3000]) * 1e-6
    c_3_data = []
    
    for t_3 in t_3_data:
        fpath = f"{_DATA_DIR}/{fname_base}{round(t_3*1e6)}us.json"
        with open(fpath) as f:
            data = json.load(f)
        c_3_data.append(np.abs(data["batman_popt"][-1]))
    c_3_data = np.array(c_3_data)

    # fit contrast decay
    # c_1_popt, _ = curve_fit(contrast_decay, t_1_data, c_1_data)
    # c_1_fit = contrast_decay(t_1_plot, *c_1_popt)
    c_3_popt, _ = curve_fit(contrast_decay, t_3_data, c_3_data, p0=[max(c_3_data), t_3_data[-1]])
    c_3_fit = contrast_decay(t_3_plot, *c_3_popt)

    # sens_1 = c_1_fit * (t_1_plot * 1e6)
    sens_3 = c_3_fit * (t_3_plot * 1e6)

    # ── Figure layout ──────────────────────────────────────────────────────────
    if vertical:
        fig = plt.figure(figsize=(6.5, 9))
        gs_outer = GridSpec(2, 1, height_ratios=[3, 2.2], hspace=0.35,
                            left=0.13, right=0.87, top=0.95, bottom=0.08)
        gs_inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0], hspace=0.0)
        ax_p1 = fig.add_subplot(gs_inner[0])
        ax_p2 = fig.add_subplot(gs_inner[1], sharex=ax_p1)
        ax_p3 = fig.add_subplot(gs_inner[2], sharex=ax_p1)
        ax    = fig.add_subplot(gs_outer[1])
    else:
        fig = plt.figure(figsize=(11, 4.5))
        gs = GridSpec(3, 2, width_ratios=[2.5, 1.8], wspace=0.25, hspace=0.03,
                      left=0.07, right=0.96, top=0.93, bottom=0.13)
        ax_p1 = fig.add_subplot(gs[0, 0])
        ax_p2 = fig.add_subplot(gs[1, 0], sharex=ax_p1)
        ax_p3 = fig.add_subplot(gs[2, 0], sharex=ax_p1)
        ax    = fig.add_subplot(gs[:, 1])

    phase_axes = [ax_p1, ax_p2, ax_p3]

    # ── Phase scan panels ──────────────────────────────────────────────────────


    KDE_WIDTH = 1.5
    X_RIGHT   = 4 * np.pi + 0.3
    X_LIM     = X_RIGHT + KDE_WIDTH + 0.2
    phi_fit = np.linspace(0, 4*np.pi, 1000)
    for i, (axi, t_scan, lbl, col, mkr) in enumerate(
            zip(phase_axes, scan_times, scan_labels, scan_colors, scan_markers)):
        fpath = f"{_DATA_DIR}/{fname_base}{int(t_scan*1e6)}us.json"
        with open(fpath) as f:
            data = json.load(f)

        phase = data["phases"]
        P_data = data["xc_avg"]
        P_std = data["xc_std"]
        popt = data["sine_popt"]
        popt2 = data["batman_popt"]


        P_fit = fringe(phi_fit, *popt)

        axi.errorbar(phase, P_data, P_std, fmt=mkr, color=col,
                 markersize=4.5, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        axi.plot(phi_fit, P_fit, '--', color='black', linewidth=LW_FIT, zorder=5)

        kde = gaussian_kde(data["xc"], bw_method=0.3)
        y_grid = np.linspace(-0.1, 1.1, 300)
        density = kde(y_grid)
        density_scaled = X_RIGHT + (density / density.max()) * KDE_WIDTH

        axi.fill_betweenx(y_grid, X_RIGHT, density_scaled,
                           color=col, alpha=0.3, zorder=2)
        axi.plot(density_scaled, y_grid,
                 color=col, linewidth=1.0, alpha=0.7, zorder=2)


        y_max = popt2[1] + popt2[2]/2
        y_min = popt2[1] - popt2[2]/2
        ## this discrepency between histogram and density is giving offset but can resolve later
        if i == 0:
            # Double-headed arrow spanning y_min → y_max, labeled C, placed within the hlines
            x_arr = X_RIGHT + 0.17
            axi.annotate('', xy=(x_arr, y_min), xytext=(x_arr, y_max),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.2),
                         zorder=7)
            axi.text(X_RIGHT + 0.35, (y_min + y_max) / 2, r'$C$',
                     ha='left', va='center', fontsize=FS_CORNER + 2, color='black', zorder=7)

        for y_val in [y_min, y_max]:
            axi.hlines(y_val, X_RIGHT, X_RIGHT + 0.3,
                       color='black', linewidth=1.2, zorder=5)

        axi.axvline(X_RIGHT, color='#cccccc', linewidth=0.6, zorder=1)

        axi.set_xlim(-0.5, X_LIM)
        axi.set_ylim(-0.1, 1.1)
        axi.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        axi.set_yticks([0, 0.5, 1])
        axi.set_yticklabels(['0', '0.5', '1'], fontsize=FS_TICK)
        axi.tick_params(axis='both', labelsize=FS_TICK)

        if i == 1:
            axi.set_ylabel(r'$^3P_0$ Population', fontsize=FS_LABEL)

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
    lx = -0.13 if vertical else -0.18
    add_panel_label(ax_p1, 'a)', x=lx)
    add_panel_label(ax, 'b)')

    # ── Right panel: Contrast decay ────────────────────────────────────────────
    
    # ax.scatter(t_1_data * 1e6, c_1_data, s=60, marker='o', color=COLOR_1V,
    #         ec='white', linewidth=1, zorder=4, label='1-photon')
    # ax.plot(t_1 * 1e6, c_1, color=COLOR_1V, linewidth=LW_MAIN, zorder=3)
    ax.scatter(t_3_data * 1e6, c_3_data, s=60, marker='s', color=COLOR_3V,
            ec='white', linewidth=1, zorder=4, label='3-photon')
    ax.plot(t_3_plot * 1e6, c_3_fit, color=COLOR_3V, linewidth=LW_MAIN, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Ramsey delay $T$ ($\mu$s)', fontsize=FS_LABEL)
    ax.set_ylabel('Contrast', fontsize=FS_LABEL, labelpad=2)
    ax.set_xlim(0.7, 4e4)
    ax.set_ylim(4e-3, 3)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['1', '10', '100', r'$10^3$', r'$10^4$'], fontsize=FS_TICK)

    ax.set_yticks([1e-2, 0.1, 1])
    ax.set_yticklabels(['0.01', '0.1', '1'], fontsize=FS_TICK)
    ax.legend(frameon=False, fontsize=FS_TICK, loc='upper left',
              handlelength=1.5, handletextpad=0.5)

    # ── Twin axis: Sensitivity C·T ─────────────────────────────────────────────
    ax2 = ax.twinx()
    # max_sens1 = max(sens_1)
    max_sens1=1
    # ax2.plot(t_1 * 1e6, sens_1/max_sens1, '--', color=COLOR_1V_LIGHT, linewidth=LW_AUX, zorder=2)
    ax2.plot(t_3_plot * 1e6, sens_3/max_sens1, '--', color=COLOR_3V_LIGHT, linewidth=LW_AUX, zorder=2)
    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 2e3)
    ax2.set_ylabel(r'Relative Sensitivity  $\propto C \cdot T$  ($\mu$s)',
                   fontsize=FS_LABEL, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_yticks([0.1, 1, 10, 100, 1000])
    ax2.set_yticklabels(['0.1', '1', '10', '100', r'$10^3$'],
                        fontsize=FS_TICK, color='gray')
    

    ax2.axhline(max(sens_3/max_sens1), xmin=0.6, xmax=0.85, linestyle=':', color='black', lw=2)
    print(max(sens_3/max_sens1))
    ax2.text(0.5 + 0.45/2, 0.88, "~1000x Enhancement", fontsize=10, 
             transform=ax2.transAxes, ha='center')
    return fig


if __name__ == '__main__':
    fig = make_figure()
    suffix = "v" if VERTICAL else "h"
    save_figure(fig, f'fig4{suffix}')
    plt.show()
