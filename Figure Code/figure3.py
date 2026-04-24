#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from pdf2image import convert_from_path


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


def load_pdf_as_array(pdf_path, dpi=600):
    """Rasterize first page of a PDF and return as a numpy array."""
    pil_img = convert_from_path(pdf_path, dpi=dpi)[0]
    return np.array(pil_img)


def make_figure():
    fig = plt.figure(figsize=(13,3))

    # Image placement (oversized axes — intentional for full-bleed image)
    #ax_img = fig.add_axes([0.04, -0.3, 0.57, 1.6])

    # Plot gridspec
    gs = GridSpec(1,3, hspace=0.01, wspace=0.12, left=0.05, right=0.98, top=0.94, bottom=0.18)
    ax_p1 = fig.add_subplot(gs[0])
    ax_p2 = fig.add_subplot(gs[1])
    ax_p3 = fig.add_subplot(gs[2])

    # ── Sequence image ─────────────────────────────────────────────────────────
    #img = load_pdf_as_array(str(_FIGURE_DIR / 'doppler_erasing.pdf'))
    #ax_img.imshow(img)
    #ax_img.axis('off')

    # ── Panel labels ───────────────────────────────────────────────────────────
    # 'a)' uses fig.text because ax_img has non-standard coordinates
    #fig.text(0.03, 0.95, 'a)', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
    add_panel_label(ax_p1, 'a)', x=-0.08, y=1.05)
    add_panel_label(ax_p2, 'b)', x=-0.08,y=1.05)
    add_panel_label(ax_p3, 'c)', x=-0.08, y=1.05)

    # ──  data ──────────────────────────────────────────────────────────────
    #region
    data_1 = np.loadtxt(str(_DATA_DIR / 'seq_pi2.csv'), delimiter=',')
    t_raw_1 = data_1[0,:]
    pop_1s0_1 = data_1[1,:]
    pop_3p1_1 = data_1[2,:]
    pop_3p0_1= data_1[3,:]

    data_2 = np.loadtxt(str(_DATA_DIR / 'seq_pi_040226.csv'), delimiter=',')
    t_raw_2 = data_2[0,:]
    pop_1s0_2 = data_2[1,:]
    pop_3p1_2 = data_2[2,:]
    pop_3p0_2= data_2[3,:]

    data_3 = np.loadtxt(str(_DATA_DIR / 'SeqRabi3P0.csv'), delimiter=',',skiprows=2)
    t_raw_3 = data_3[:,0]
    pop_1s0_3 = data_3[:,1]
    pop_3p1_3 = data_3[:,2]
    pop_3p0_3= data_3[:,3]
    #endregion

    #rabi_color = [230/255, 230/255, 230/255]
    #raman_color = [215/255,215/255,215/255]


    # ── Method 1 panel ─────────────────────────────────────────────────────────
    ax_p1.set_ylabel('Population', fontsize=FS_LABEL, labelpad=2)
    ax_p1.set_ylim([-0.05, 1.05])
    ax_p1.set_xlim([-0.05, 0.65])
    #ax_p1.set_xticks([])
    ax_p1.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)

    ax_p1.scatter(t_raw_1, pop_1s0_1,
                  s=35, marker='o', ec='k', color=COLOR_1S0, zorder=6)
    ax_p1.scatter(t_raw_1, pop_3p1_1,
                  s=35, marker='^', ec='k', color=COLOR_3P1, zorder=6)
    ax_p1.scatter(t_raw_1, pop_3p0_1,
                  s=35, marker='s', ec='k', color=COLOR_3P0, zorder=6)

    ax_p1.plot([], [], marker='o', linestyle='--', label=r'$^1S_0$', color=COLOR_1S0,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    ax_p1.plot([], [], marker='s', linestyle='--', label=r'$^3P_0$', color=COLOR_3P0,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    ax_p1.plot([], [], marker='^', linestyle='--', label=r'$^3P_1$', color=COLOR_3P1,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    # ax_p1.plot([], [], marker='*', linestyle='--', label=r'$^3P_2$', color=COLOR_3P2,
    #            markeredgewidth=0.8, markeredgecolor='black', markersize=9)

    #ax_p1.axvspan(0,0.075, color=rabi_color, alpha=1)
    #ax_p1.axvspan(0.075,0.57, color=raman_color, alpha=1)
    #ax_p1.axvline(x=0.075, color='k', alpha=.5)
    ax_p1.set_xlabel(r'Sequence Duration (μs)', fontsize=FS_LABEL)

    tpi2 = 0.06
    # ax_p1.axvspan(-.02, tpi2, facecolor='g', alpha=0.2)

    # ax_p1.axvspan(tpi2, 0.64, facecolor='b', alpha=0.2)



    # ── Method 2 panel ─────────────────────────────────────────────────────────
    #ax_p2.set_ylabel('Population', fontsize=FS_LABEL, labelpad=2)
    ax_p2.set_yticks([])
    ax_p2.set_ylim([-0.05, 1.05])
    ax_p2.set_xlim([-0.05, 0.65])
    ax_p2.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)


    ax_p2.scatter(t_raw_2, pop_1s0_2,
                  s=35, marker='o', ec='k', color=COLOR_1S0, zorder=6)
    ax_p2.scatter(t_raw_2, pop_3p1_2,
                  s=35, marker='^', ec='k', color=COLOR_3P1, zorder=6)
    ax_p2.scatter(t_raw_2, pop_3p0_2,
                  s=35, marker='s', ec='k', color=COLOR_3P0, zorder=6)

    ax_p2.set_xlabel(r'Sequence Duration (μs)', fontsize=FS_LABEL)

    ax_p2.annotate('', xy=(0.605, 0.005), xytext=(0.605, .98),
                   arrowprops=dict(arrowstyle='<|-|>, head_width=0.2, head_length=.6',
                                   color='black', lw=.7),
                   zorder=6)
    ax_p2.text(0.6, 0.5, r'95.4%',
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               color='black', zorder=6)

    # ── Method 3 panel ─────────────────────────────────────────────────────────
    #ax_p3.set_ylabel('Population', fontsize=FS_LABEL, labelpad=2)
    ax_p3.set_yticks([])
    ax_p3.set_ylim([-0.05, 1.05])
    ax_p3.set_xlim([-0.05, 0.65])
    ax_p3.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)


    ax_p3.scatter(t_raw_3, pop_1s0_3,
                  s=35, marker='o', ec='k', label=r'$^1S_0$',color=COLOR_1S0, zorder=6)
    ax_p3.scatter(t_raw_3, pop_3p1_3,
                  s=35, marker='^', ec='k', label=r'$^3P_1$',color=COLOR_3P1, zorder=6)
    ax_p3.scatter(t_raw_3, pop_3p0_3,
                  s=35, marker='s', ec='k',label=r'$^3P_0$', color=COLOR_3P0, zorder=6)

    ax_p3.set_xlabel(r'Sequence Duration (μs)', fontsize=FS_LABEL)
    ax_p3.legend(loc='upper right', fontsize=10, frameon=False, ncol=1,labelspacing=0.2,
                    columnspacing=0.2, handlelength=0, handletextpad=0.5,bbox_to_anchor=(0.98,1.04))

    #ax_p3.annotate('', xy=(0.605, 0.005), xytext=(0.605, .98),
    #               arrowprops=dict(arrowstyle='<|-|>, head_width=0.2, head_length=.6',
    #                               color='black', lw=.7),
    #               zorder=6)
    #ax_p3.text(0.6, 0.5, r'95.4%',
    #          ha='right', va='bottom', fontsize=10, fontweight='bold',
    #           color='black', zorder=6)
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig3')
    plt.show()
