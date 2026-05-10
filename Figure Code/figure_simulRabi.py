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

def gaussian(f, f0, sigma, A):
    return A / np.sqrt(2*PI*np.abs(sigma)) * np.exp(-(f-f0)**2/(2*sigma**2))


def make_figure():
    data_ro1 = np.loadtxt(_DATA_DIR / 'Rabi1.csv', delimiter=',', skiprows=0)
    data_ro2 = np.loadtxt(_DATA_DIR / 'Rabi2.csv', delimiter=',', skiprows=0)

    data_err1 = np.loadtxt(_DATA_DIR / 'Rabi1err.csv', delimiter=',', skiprows=0)
    data_err2 = np.loadtxt(_DATA_DIR / 'Rabi2err.csv', delimiter=',', skiprows=0)


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

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6, 4))
    gs  = GridSpec(1, 1, hspace=0.22, left=0.13, right=0.95, top=0.96, bottom=0.07)
    ax2 = fig.add_subplot(gs[0])

    # ──  Rabi flopping ───────────────────────────────────────────────
    ax2.errorbar(t_data, pop_3P0_data, yerr = pop_3P0_err, color=COLOR_3P0, fmt='s', markeredgecolor='black', markersize = 8)
    ax2.errorbar(t_data, pop_3P1_data, yerr = pop_3P1_err, color=COLOR_3P1, fmt='^', markeredgecolor='black', markersize = 9)
    ax2.errorbar(t_data, pop_3P2_data, yerr = pop_3P2_err, color=COLOR_3P2, fmt='*', markeredgecolor='black', markersize = 10)

    # Simulation data
    with open(_DATA_DIR / "sim_pops_v2.txt", 'r') as f:
        header = f.readline().strip()
        sim_data = np.loadtxt(f, delimiter=',')

    t_sim = sim_data[:,0]
    pop_1S0_sim = sim_data[:,1]
    pop_3P1_sim = sim_data[:,2] + sim_data[:,3]
    pop_3P2_sim = sim_data[:,5]
    pop_3P0_sim = sim_data[:,6]

    ax2.plot(t_sim, pop_3P0_sim, linestyle='--', color=COLOR_3P0)
    ax2.plot(t_sim, pop_3P1_sim, linestyle='--', color=COLOR_3P1)
    ax2.plot(t_sim, pop_3P2_sim, linestyle='--', color=COLOR_3P2)
        
    ax2.plot([], [], linestyle='--', label="Simulation", color='k',
            linewidth=2)
    ax2.scatter([], [], marker='s', label=r'$^3P_0$', color=COLOR_3P0, ec='k', s=70)
    ax2.scatter([], [], marker='^',label=r'$^3P_1$', color=COLOR_3P1, ec='k', s=70)
    ax2.scatter([], [], marker='*', label=r'$^3P_2$', color=COLOR_3P2, ec='k', s=70)
    

    
    ax2.set_xlabel(r'Pulse duration ($\mathrm{\mu}$s)', fontsize=FS_LABEL)
    ax2.set_ylabel('Population', fontsize=FS_LABEL)
    ax2.set_xlim(0 - 0.25, max(t_data) + 0.25)
    ax2.set_ylim(-0.05, 1.05)

    ax2.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)
    ax2.legend(loc='upper left', fontsize=10, frameon=False, ncol=1,
               columnspacing=0.5, handlelength=3, handletextpad=0.25)
    
    #rect2 = patches.Rectangle((9.85, -0.03), .4, .3, linewidth=1.5, edgecolor='k', facecolor='none')
    ax2.plot([7.8, 9], [.71, .75], c='k', lw=1.5, solid_capstyle='round')

    # Inset: atom cloud image
    width=0.5
    height = 0.4
    img = plt.imread(str(_DATA_DIR / f'cloud_images.png'))
    axins_img = ax2.inset_axes([0.59, 0.63, width, height])
    axins_img.set_xticks([])
    axins_img.set_yticks([])
    axins_img.imshow(img)



    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig_simulRabi')
    plt.show()

# %%
