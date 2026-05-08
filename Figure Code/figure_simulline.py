#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import patheffects
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


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

def exp_sine(t, A, w, tau):
    return (A*np.sin(np.pi*w*t)**2 * np.exp(-t/tau))



# ── Figure-specific marker shapes ─────────────────────────────────────────────
MARKER_3V = 's'
MARKER_1V = 'o'

# ── Physical constants ────────────────────────────────────────────────────────
c  = 3e8
m  = 88 * 1.66e-27
kb = 1.38e-23
PI = np.pi
v0 = 2 * PI * 434e12

# ── Fit functions ─────────────────────────────────────────────────────────────
def lorentzian(f, f0, gamma, A):
    return A / (1 + ((f - f0) / (gamma / 2))**2)

def lorentzian_offset(f, f0, gamma, A,y0):
    return (A / (1 + ((f - f0) / (gamma / 2))**2)+y0)

def gaussian(f, f0, sigma, A):
    return A / np.sqrt(2*PI*np.abs(sigma)) * np.exp(-(f-f0)**2/(2*sigma**2))

def gaussian_offset(f, f0, sigma, A,y0):
    return (A / np.sqrt(2*PI*np.abs(sigma)) * np.exp(-(f-f0)**2/(2*sigma**2))+y0)


def make_figure():
    # ── Panel (a) data ─────────────────────────────────────────────────────────
    _scan_1v   = np.loadtxt(_DATA_DIR / 'simulline_1v_scan.txt', comments='#')
    f_1v_MHz   = _scan_1v[:, 0]
    y_1v       = _scan_1v[:, 1]

    _scan_3v   = np.loadtxt(_DATA_DIR / 'simulline_3v_scan.txt', comments='#')
    f_3v_MHz   = _scan_3v[:, 0]
    y_3v       = _scan_3v[:, 1]

    y_3v_norm = y_3v / np.max(y_3v)
    y_1v_norm = y_1v / np.max(y_1v)

    popt_3v, _ = curve_fit(lorentzian, f_3v_MHz, y_3v_norm, p0=[-30, 10, 1.0], maxfev=10000)
    popt_1v, _ = curve_fit(gaussian,   f_1v_MHz, y_1v_norm, p0=[0, 50, 1.0],  maxfev=10000)

    f_3v_shifted = f_3v_MHz - popt_3v[0]
    f_1v_shifted = f_1v_MHz - popt_1v[0]
    f_fine = np.linspace(-100, 100, 1000)
    y_3v_fit = lorentzian(f_fine, 0, popt_3v[1], popt_3v[2])
    y_1v_fit = gaussian(f_fine,   0, popt_1v[1], popt_1v[2])

    T_uk = 7
    vdopp_rad = v0 * np.sqrt(8 * kb * T_uk*1e-6 * np.log(2) / m / c**2)
    vdopp = vdopp_rad / (2 * PI)

    _lw_1v     = np.loadtxt(_DATA_DIR / 'simulline_lw_1v.txt', comments='#')
    pi_times1  = _lw_1v[:, 0]
    omega_1v   = (PI / pi_times1) / (2*PI)
    lw_1v_khz  = _lw_1v[:, 1]
    lw_1v_err  = _lw_1v[:, 2]

    _lw_3v     = np.loadtxt(_DATA_DIR / 'simulline_lw_3v.txt', comments='#')
    pi_times3  = _lw_3v[:, 0]
    omega_3v   = (PI / pi_times3) / (2*PI)
    lw_3v_khz  = _lw_3v[:, 1]
    lw_3v_err  = _lw_3v[:, 2]


    rabi_freq_rad = 2 * PI * np.logspace(2, 7.2, 500)
    rabi_freq     = rabi_freq_rad / (2*PI)
    f_limit3      = 2 * 0.89 * rabi_freq
    lw_1v_theory  = np.sqrt(f_limit3**2 + vdopp**2)
    lw_3v_theory  = f_limit3

    # # ── Panel (b) data ─────────────────────────────────────────────────────────
    _narrow      = np.loadtxt(_DATA_DIR / 'simulline_narrow_scan.txt', comments='#')
    f_narrow     = _narrow[:, 0]
    pop_narrow   = _narrow[:, 1]

    popt_narrow, pcov_narrow = curve_fit(lorentzian_offset, f_narrow, pop_narrow, p0=[0, 250, 1.0,0.05], maxfev=10000)
    #popt_narrow, pcov_narrow = curve_fit(gaussian_offset, f_narrow, pop_narrow, p0=[0, 100, 1.0,0.05], maxfev=10000)
    

    # ── Panel (b) data ─────────────────────────────────────────────────────────
    #data_ro1 = np.loadtxt(_DATA_DIR / 'RabiFloppingData1.txt', delimiter=',', skiprows=2)
    #data_ro2 = np.loadtxt(_DATA_DIR / 'RabiFloppingData2.txt', delimiter=',', skiprows=2)
    data_ro1 = np.loadtxt(_DATA_DIR / 'Rabi1.csv', delimiter=',', skiprows=2)
    data_ro2 = np.loadtxt(_DATA_DIR / 'Rabi2.csv', delimiter=',', skiprows=2)

    data_err1 = np.loadtxt(_DATA_DIR / 'Rabi1err.csv', delimiter=',', skiprows=2)
    data_err2 = np.loadtxt(_DATA_DIR / 'Rabi2err.csv', delimiter=',', skiprows=2)


    assert np.all(data_ro1[:, 0] == data_ro2[:, 0])
    t_data       = data_ro1[:, 0]
    pop_1S0_data = data_ro1[:, 1]
    pop_3P1_data = data_ro1[:, 2]
    pop_3P2_data = 4/3 * data_ro2[:, 2]
    pop_3P0_data = (0.5 * (data_ro1[:, 3] - pop_3P2_data)
                  + 0.5 * (data_ro2[:, 3] - 0.25 * pop_3P2_data))
    
    pop_1S0_err = data_err1[:, 1]
    pop_3P1_err = data_err1[:, 2]
    pop_3P2_err = 4/3 * (data_err2[:, 2])
    pop_3P0_err = 0.5 * (data_err1[:, 3])+ 0.5*(data_err2[:, 3])

    popt, pcov = curve_fit(exp_sine, t_data, pop_3P0_data, sigma=(pop_3P0_err/pop_3P0_data), p0=[0.8,0.2,20], maxfev=20000)


    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4,4))
    gs  = GridSpec(1, 2, hspace=0.22, wspace=0.32, left=-0.25, right=1.25, top=0.96, bottom=0.07,width_ratios=[2.2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Panel (a): Linewidth vs Rabi ───────────────────────────────────────────
    ax1.scatter(omega_1v / 1e3, lw_1v_khz,
                marker=MARKER_1V, ec='k', fc=COLOR_1V, s=MARKER_S, zorder=2)
    #ax1.errorbar(omega_1v / 1e3, lw_1v_khz, yerr = lw_1v_err, color=COLOR_1V, fmt='o', alpha=0.8,markeredgecolor='black', markersize = 8)
    
    ax1.scatter(omega_3v / 1e3, lw_3v_khz,
                marker=MARKER_3V, ec='k', fc=COLOR_3V, s=MARKER_S, zorder=2)

    ax1.axhline(vdopp / 1e3 - 7, xmin=0.02, xmax=0.98, color=COLOR_DLIM, linestyle='--', zorder=0)
    ax1.text(0.075, 45, r"Doppler Limit [7$\,\mathrm{\mu}$K]", c=COLOR_DLIM, fontsize=11)

    ax1.plot(rabi_freq / 1e3, lw_3v_theory / 1e3,
             color=COLOR_3V, linewidth=LW_MAIN, linestyle='-', zorder=1)
    ax1.plot(rabi_freq / 1e3, lw_1v_theory / 1e3,
             color=COLOR_1V, linewidth=LW_MAIN, linestyle='-', zorder=1)

    legend_marker_size = 8
    ax1.plot([], [], linestyle='-', marker=MARKER_3V, markersize=legend_marker_size,
             markeredgecolor='k', markerfacecolor=COLOR_3V, label='3-Photon', color=COLOR_3V)
    ax1.plot([], [], linestyle='-', marker=MARKER_1V, markersize=legend_marker_size,
             markeredgecolor='k', markerfacecolor=COLOR_1V, label='1-Photon', color=COLOR_1V)

    ax1.set_xlabel('Rabi Frequency (kHz)', fontsize=FS_LABEL)
    ax1.set_ylabel('Linewidth (kHz)', fontsize=FS_LABEL)
    ax1.set_xlim(0.05, 1.5e3)
    ax1.set_ylim(0.1, 2e3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([0.1, 1, 10, 1e2, 1e3])
    ax1.set_xticklabels([0.1, 1, 10, r'$10^2$', r'$10^3$'])
    ax1.set_yticks([1, 10, 1e2, 1e3])
    ax1.set_yticklabels([1, 10, r'$10^2$', r'$10^3$'])
    ax1.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)
    ax1.legend(loc='lower right', fontsize=10, frameon=False, handlelength=2.5, handletextpad=0.2)

    #rect = patches.Rectangle((4.6, 5), 5, 170, linewidth=1.5, edgecolor='k', facecolor='none')
    #ax1.add_patch(rect)
    #ax1.plot([2, 7], [1e3, 178], c='k', lw=1.5, solid_capstyle='round')
    #ax1.plot([10, 200], [70, 15], c='k', lw=1.5, solid_capstyle='round')
    
    # Inset: frequency scans
    # axins = ax1.inset_axes([0.625, 0.15, 0.35, 0.4])
    # axins.scatter(f_1v_shifted, y_1v_norm, marker=MARKER_1V, color=COLOR_1V, alpha=0.7, s=12)
    # axins.plot(f_fine, y_1v_fit, color=COLOR_1V, linewidth=1, linestyle=':')
    # axins.scatter(f_3v_shifted, y_3v_norm, marker=MARKER_3V, color=COLOR_3V, alpha=0.7, s=12)
    # axins.plot(f_fine, y_3v_fit, color=COLOR_3V, linewidth=1, linestyle=':')
    # axins.set_yticks([])
    # axins.set_xlabel(r'$\Delta\nu$ (kHz)', fontsize=8, labelpad=0)
    # axins.set_ylabel(r'${}^3P_0$ population', fontsize=8, labelpad=0)
    # axins.tick_params(axis='x', labelsize=8)
    # axins.set_xlim(-90, 90)
    # axins.set_ylim(-0.1, 1.2)

    ############# INSET HERE #######
    # axins = ax1.inset_axes([0.565, 0.11, 0.35, 0.4])
    # axins.scatter(f_3v_shifted, y_3v, marker=MARKER_3V, color=COLOR_3V,ec='k', alpha=0.7, s=12)
    # axins.plot(f_fine, y_3v_fit*max(y_3v), color=COLOR_3V, linewidth=1, linestyle=':')
    # axins.set_yticks([0,1])
    # axins.set_xlabel(r'$\Delta\nu$ (kHz)', fontsize=8, labelpad=0)
    # axins.set_ylabel(r'${}^3P_0$ population', fontsize=8, labelpad=-5, color=COLOR_3V)
    # axins.tick_params(axis='x', labelsize=8)
    # axins.tick_params(axis='y', labelsize=8)
    # axins.tick_params(axis='y',colors=COLOR_3V, labelsize=8)
    # axins.spines['left'].set_color(COLOR_3V)
    # axins.set_xlim(-90, 90)
    # axins.set_ylim(-0.1, 1.05)
    
    # axins2 = axins.twinx()
    # axins2.scatter(f_1v_shifted, y_1v, marker=MARKER_1V, color=COLOR_1V, alpha=0.7, s=12)
    # axins2.plot(f_fine, y_1v_fit*max(y_1v), color=COLOR_1V, linewidth=1, linestyle=':')
    # axins2.set_yticks([0,0.04])
    # axins2.set_ylabel(r'${}^3P_1$ Population',fontsize=8, labelpad=-10, color=COLOR_1V)
    # axins2.tick_params(axis='y',colors=COLOR_1V, labelsize=8)
    # axins2.set_ylim(-0.005, 0.045)
    # axins2.spines['right'].set_color(COLOR_1V)
    ######################
    # axins.patch.set_path_effects([
    #     patheffects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.7)
    # ])

    ax1.scatter((PI / 3000) / (2*PI)*10**3, 0.286, marker='*', color='gold', s=400, edgecolor='k')
    

    add_panel_label(ax1, 'a)',x=-0.2,y=1.05)
    # ── Panel (b): Rabi flopping ───────────────────────────────────────────────
    #ax2.plot(t_data, 0.1*t_data/20, linestyle='--', color=COLOR_3P2)
    #ax2.plot(t_data, 0.05*t_data/40, linestyle='--', color=COLOR_3P1)
    #ax2.plot(t_data, (0.5 + 0.5*np.cos(2*np.pi * t_data/6))*np.exp(-t_data/35), linestyle='--', color=COLOR_1S0)
    #ax2.plot(t_data, (0.5 - 0.5*np.cos(2*np.pi * t_data/6))*np.exp(-t_data/35), linestyle='--', color=COLOR_3P0)
    #ax2.plot(t_data, exp_sine(t_data,*popt), linestyle='-', color=COLOR_3P0)
    
    #ax2.plot(t_data, (0.5 - 0.5*np.cos(2*np.pi * t_data*0.197)), linestyle='--', color='black', label = 'Theory (Eq. 1)')

    #ax2.scatter(t_data, pop_1S0_data, s=MARKER_S, marker='s', ec='k', color=COLOR_1S0)
    #ax2.errorbar(t_data, pop_3P0_data, yerr = pop_3P0_err, color=COLOR_3P0, fmt='s', alpha=0.8,markeredgecolor='black', markersize = 8)
    #ax2.errorbar(t_data, pop_3P1_data, yerr = pop_3P1_err, color=COLOR_3P1, fmt='^', alpha=0.8,markeredgecolor='black', markersize = 9)
    #ax2.errorbar(t_data, pop_3P2_data, yerr = pop_3P2_err, color=COLOR_3P2, fmt='*', alpha=0.8,markeredgecolor='black', markersize = 10)
    #ax2.errorbar(np.delete(t_data,32), np.delete(pop_3P2_data,32), yerr = np.delete(pop_3P2_err,32), color=COLOR_3P2, fmt='*', alpha=0.8,markeredgecolor='black', markersize = 10)
    
    #print(max(np.delete(pop_3P2_data,32)))
    #ax2.scatter(t_data, pop_3P0_data,s=MARKER_S, marker='s', ec='k', color=COLOR_3P0)
    #ax2.scatter(t_data, pop_3P1_data, s=35, marker='^', ec='k', color=COLOR_3P1, alpha=0.8)
    #ax2.scatter(t_data, pop_3P2_data, s=35, marker='*', ec='k', color=COLOR_3P2, alpha=0.8)
    ax2.scatter(f_narrow,pop_narrow,s=MARKER_S,color=COLOR_3V, alpha=0.7, marker=MARKER_3V, ec='k')
    #ax2.plot([], [], marker='s', linestyle='--', label=r'$^1S_0$', color=COLOR_1S0,
    #         markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    #ax2.plot([], [], marker='s', linestyle='-', label=r'$^3P_0$', color=COLOR_3P0,
    #         markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    ax2.plot(f_narrow,lorentzian_offset(f_narrow,*popt_narrow),color=COLOR_3V, alpha=0.7)
    # ax2.plot([], [], marker='^', linestyle='--', label=r'$^3P_1$', color=COLOR_3P1,
    #          markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    # ax2.plot([], [], marker='*', linestyle='--', label=r'$^3P_2$', color=COLOR_3P2,
    #          markeredgewidth=0.8, markeredgecolor='black', markersize=10)
    # print(popt_narrow)
    
    ax2.set_xlabel(r'$\Delta_3/2\pi$ (Hz)', fontsize=FS_LABEL)
    ax2.set_ylabel(r'${}^3P_0$ Population', fontsize=FS_LABEL)
    ax2.set_xlim(-1500, 1500)
    ax2.set_ylim(-0.05, 0.85)
    ax2.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)
    # ax2.legend(loc='upper left', fontsize=10, frameon=False, ncol=1,
    #            columnspacing=0.5, handlelength=3, handletextpad=0.25)
    

    ax2.scatter(1100,0.8, marker='*', color='gold', s=400, edgecolor='k')
    
    # print(np.sqrt(np.diag(pcov_narrow)))

    add_panel_label(ax2, 'b)',x=-0.4,y=1.05)

    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig_simulline')
    plt.show()

# %%
