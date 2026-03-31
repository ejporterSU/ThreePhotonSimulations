#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import patheffects
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
import os
os.chdir(r"C:/Users/Erik\Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code")
from fig_style import *
_DATA_DIR = Path(__file__).parent.parent / 'Data'


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

def gaussian(f, f0, sigma, A):
    return A / np.sqrt(2*PI*np.abs(sigma)) * np.exp(-(f-f0)**2/(2*sigma**2))


def make_figure():
    # ── Panel (a) data ─────────────────────────────────────────────────────────
    f_1v_MHz = np.array([-180., -166.66666667, -153.33333333, -140., -126.66666667,
        -113.33333333, -100., -86.66666667, -73.33333333, -60., -46.66666667,
        -33.33333333, -20., -6.66666667, 6.66666667, 20., 33.33333333,
        46.66666667, 60., 73.33333333, 86.66666667, 100., 113.33333333,
        126.66666667, 140., 153.33333333, 166.66666667, 180., 193.33333333,
        206.66666667, 220.])

    y_1v = np.array([0.0011168508763745042, 0.0018965412083842804, 0.0012552663042992125,
        0.0012686411130136018, 0.0008040376797447392, 0.0011158087434075753,
        0.002974408809404286, 0.0013864253800040414, 0.0015618119367055163,
        0.002189493228789459, 0.006793031982768895, 0.01183841913425912,
        0.023202505682133157, 0.028498858576497976, 0.03566167919993798,
        0.030064832858641883, 0.02632607686481468, 0.016162177338900935,
        0.006424428752533944, 0.0039018274801318835, 0.001768329084627986,
        0.0005909373362738051, 0.0004298499250628297, 0.0023528202742793473,
        0.0023147837702275353, 0.002903346741345631, 0.0021572056887514427,
        0.002038369304556355, 0.002350649389097605, 0.0015792420195358088,
        0.003592909844627954])

    y_3v = np.array([0.005040271176251144, 0.0026282707369170524, 0.0011889584437536707,
        0.009708389956419749, 0.0, 0.007092358646527452,
        0.00045177128439428553, 0.003784300708605921, 0.01606494774738008,
        0.005518228939803119, 0.021699434838833277, 0.02210032594005883,
        0.0018075674325674326, 0.0005656108597285068, 0.000871324188631881,
        0.01973905524989366, 0.015223028669940748, 0.03409204941886204,
        0.09463500173844688, 0.012875241597485893, 0.009087708552177758,
        0.016841363200516175, 0.08463234895365723, 0.25152991593426305,
        0.4424884363992948, 0.7023903192281638, 0.8924532343593577,
        0.8723758889496718, 0.6332810122335518, 0.45822297217861563,
        0.31914666355938787, 0.15789038078321269, 0.06533490011750881,
        0.030497013747188752, 0.05516880005988472, 0.06014401824752723,
        0.04740793506481334, 0.03146662895753727, 0.036369123525426136,
        0.008459152700942479, 0.018474667531249458, 0.013274178604518566,
        0.014063343883607048, 0.01142562288073124, 0.02078879688405905,
        0.007335763574015242, 0.003325540710006038, 0.010791693116894895,
        0.011151692825146201, 0.007645931672664392, 0.005432471156021416])

    f_3v_MHz = np.array([-80., -78., -76., -74., -72., -70., -68., -66., -64., -62., -60.,
        -58., -56., -54., -52., -50., -48., -46., -44., -42., -40., -38.,
        -36., -34., -32., -30., -28., -26., -24., -22., -20., -18., -16.,
        -14., -12., -10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10., 12.,
        14., 16., 18., 20.])

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

    pi_times1  = np.array([1, 2, 4, 6, 9.5, 22, 38, 75]) * 1e-6
    omega_1v   = (PI / pi_times1) / (2*PI)
    lw_1v_khz  = np.array([323, 146, 92.7, 66.2, 58.5, 52.9, 47.9, 45.3]) * 2.355

    pi_times3  = np.array([2, 4, 6, 9.5, 22, 38, 75, 249, 900, 3000, 5000]) * 1e-6
    omega_3v   = (PI / pi_times3) / (2*PI)
    lw_3v_khz  = np.array([328, 187, 155, 77.2, 31.4, 21.8, 8.7, 2.64, 0.874, 0.247, 0.209])

    rabi_freq_rad = 2 * PI * np.logspace(2, 7.2, 500)
    rabi_freq     = rabi_freq_rad / (2*PI)
    f_limit3      = 2 * 0.89 * rabi_freq
    lw_1v_theory  = np.sqrt(f_limit3**2 + vdopp**2)
    lw_3v_theory  = f_limit3
    print(os.getcwd())
    # ── Panel (b) data ─────────────────────────────────────────────────────────
    data_ro1 = np.loadtxt('C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Data/RabiFloppingData1.txt', delimiter=',', skiprows=2)
    data_ro2 = np.loadtxt('C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Data/RabiFloppingData2.txt', delimiter=',', skiprows=2)

    assert np.all(data_ro1[:, 0] == data_ro2[:, 0])
    t_data       = data_ro1[:, 0]
    pop_1S0_data = data_ro1[:, 1]
    pop_3P1_data = data_ro1[:, 2]
    pop_3P2_data = 4/3 * data_ro2[:, 2]
    pop_3P0_data = (0.5 * (data_ro1[:, 3] - pop_3P2_data)
                  + 0.5 * (data_ro1[:, 3] - 0.25 * pop_3P2_data))

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.5, 9))
    gs  = GridSpec(2, 1, hspace=0.22, left=0.13, right=0.95, top=0.96, bottom=0.07)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Panel (a): Linewidth vs Rabi ───────────────────────────────────────────
    ax1.scatter(omega_1v / 1e3, lw_1v_khz,
                marker=MARKER_1V, ec='k', fc=COLOR_1V, s=MARKER_S, zorder=2)
    ax1.scatter(omega_3v / 1e3, lw_3v_khz,
                marker=MARKER_3V, ec='k', fc=COLOR_3V, s=MARKER_S, zorder=2)

    ax1.axhline(vdopp / 1e3 - 5, xmin=0.02, xmax=0.98, color=COLOR_DLIM, linestyle='-.', zorder=0)
    ax1.text(0.1e3, 45, rf"Doppler Limit [{T_uk}$\,\mu$K]", c=COLOR_DLIM, fontsize=11)

    ax1.plot(rabi_freq / 1e3, lw_3v_theory / 1e3,
             color=COLOR_3V, linewidth=LW_MAIN, linestyle='--', zorder=1)
    ax1.plot(rabi_freq / 1e3, lw_1v_theory / 1e3,
             color=COLOR_1V, linewidth=LW_MAIN, linestyle='--', zorder=1)

    legend_marker_size = 8
    ax1.plot([], [], linestyle='--', marker=MARKER_3V, markersize=legend_marker_size,
             markeredgecolor='k', markerfacecolor=COLOR_3V, label='3-Photon', color=COLOR_3V)
    ax1.plot([], [], linestyle='--', marker=MARKER_1V, markersize=legend_marker_size,
             markeredgecolor='k', markerfacecolor=COLOR_1V, label='1-Photon', color=COLOR_1V)

    ax1.set_xlabel('Rabi Frequency (kHz)', fontsize=FS_LABEL)
    ax1.set_ylabel('Linewidth (kHz)', fontsize=FS_LABEL)
    ax1.set_xlim(0.05, 3e3)
    ax1.set_ylim(0.1, 2e4)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([0.1, 1, 10, 1e2, 1e3])
    ax1.set_xticklabels([0.1, 1, 10, r'$10^2$', r'$10^3$'])
    ax1.set_yticks([1, 10, 1e2, 1e3, 1e4])
    ax1.set_yticklabels([1, 10, r'$10^2$', r'$10^3$', r'$10^4$'])
    ax1.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)
    ax1.legend(loc='lower right', fontsize=10, frameon=False, handlelength=2.5, handletextpad=0.2)

    rect = patches.Rectangle((4.6, 5), 5, 170, linewidth=1.5, edgecolor='k', facecolor='none')
    ax1.add_patch(rect)
    ax1.plot([2, 7], [1e3, 178], c='k', lw=1.5, solid_capstyle='round')

    # Inset: frequency scans
    axins = ax1.inset_axes([0.05, 0.67, 0.3, 0.3])
    axins.scatter(f_1v_shifted, y_1v_norm, marker=MARKER_1V, color=COLOR_1V, alpha=0.7, s=12)
    axins.plot(f_fine, y_1v_fit, color=COLOR_1V, linewidth=1, linestyle=':')
    axins.scatter(f_3v_shifted, y_3v_norm, marker=MARKER_3V, color=COLOR_3V, alpha=0.7, s=12)
    axins.plot(f_fine, y_3v_fit, color=COLOR_3V, linewidth=1, linestyle=':')
    axins.set_yticks([])
    axins.set_xlabel(r'$\Delta\nu$ (kHz)', fontsize=8, labelpad=0)
    axins.tick_params(axis='x', labelsize=8)
    axins.set_xlim(-90, 90)
    axins.set_ylim(-0.1, 1.2)
    axins.patch.set_path_effects([
        patheffects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.7)
    ])

    add_panel_label(ax1, 'a)')

    # ── Panel (b): Rabi flopping ───────────────────────────────────────────────
    ax2.plot(t_data, 0.1*t_data/10, linestyle='--', color=COLOR_3P2)
    ax2.plot(t_data, 0.05*t_data/10, linestyle='--', color=COLOR_3P1)
    ax2.plot(t_data, (0.5 + 0.5*np.cos(2*np.pi * t_data/6))*np.exp(-t_data/35), linestyle='--', color=COLOR_1S0)
    ax2.plot(t_data, (0.5 - 0.5*np.cos(2*np.pi * t_data/6))*np.exp(-t_data/35), linestyle='--', color=COLOR_3P0)

    ax2.scatter(t_data, pop_1S0_data, s=MARKER_S, marker='s', ec='k', color=COLOR_1S0)
    ax2.scatter(t_data, pop_3P0_data, s=MARKER_S, marker='^', ec='k', color=COLOR_3P0)
    ax2.scatter(t_data, pop_3P1_data, s=35, marker='o', ec='k', color=COLOR_3P1, alpha=0.8)
    ax2.scatter(t_data, pop_3P2_data, s=35, marker='*', ec='k', color=COLOR_3P2, alpha=0.8)

    ax2.plot([], [], marker='s', linestyle='--', label=r'$^1S_0$', color=COLOR_1S0,
             markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    ax2.plot([], [], marker='^', linestyle='--', label=r'$^3P_0$', color=COLOR_3P0,
             markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    ax2.plot([], [], marker='o', linestyle='--', label=r'$^3P_1$', color=COLOR_3P1,
             markeredgewidth=0.8, markeredgecolor='black', markersize=9)
    ax2.plot([], [], marker='*', linestyle='--', label=r'$^3P_2$', color=COLOR_3P2,
             markeredgewidth=0.8, markeredgecolor='black', markersize=9)

    ax2.set_xlabel(r'Pulse Duration ($\mu$s)', fontsize=FS_LABEL)
    ax2.set_ylabel('Population', fontsize=FS_LABEL)
    ax2.set_xlim(0 - 0.25, max(t_data) + 0.25)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)
    ax2.legend(loc='upper right', fontsize=10, frameon=False, ncol=2,
               columnspacing=0.5, handlelength=3, handletextpad=0.25)

    # Inset: atom cloud image
    img = plt.imread(str(_DATA_DIR / 'cloud_image.png'))
    width=0.11
    height = 0.35
    for i in range(2):
        axins_img = ax2.inset_axes([0.015 + width*i, 0.63, width, height])
        axins_img.set_xticks([])
        axins_img.set_yticks([])
        axins_img.imshow(img)
        axins_img.text(0.5, 0.02, f'Readout {i+1}', ha='center', va='bottom',
                   fontsize=6, color='#1d1c1c', weight='bold',
                   transform=axins_img.transAxes)


    add_panel_label(ax2, 'b)')

    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig2')
    plt.show()
