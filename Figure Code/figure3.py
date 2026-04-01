#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from fig_style import *
import os
from fig_style import *
from pdf2image import convert_from_path
_DATA_DIR = Path(__file__).parent.parent / 'Data'
_FIGURE_DIR = Path(__file__).parent.parent / 'Figure Code'


def load_pdf_as_array(pdf_path, dpi=600):
    """Rasterize first page of a PDF and return as a numpy array."""
    pil_img = convert_from_path(pdf_path, dpi=dpi)[0]
    return np.array(pil_img)


def make_figure():
    fig = plt.figure(figsize=(11, 4.5))

    # Image placement (oversized axes — intentional for full-bleed image)
    ax_img = fig.add_axes([0.04, -0.3, 0.57, 1.6])

    # Plot gridspec
    gs = GridSpec(2, 1, hspace=0.05, left=0.65, right=0.96, top=0.93, bottom=0.13)
    ax_p1 = fig.add_subplot(gs[0])
    ax_p2 = fig.add_subplot(gs[1])

    # ── Sequence image ─────────────────────────────────────────────────────────
    img = load_pdf_as_array(_FIGURE_DIR / 'doppler_erasing.pdf')
    ax_img.imshow(img)
    ax_img.axis('off')

    # ── Panel labels ───────────────────────────────────────────────────────────
    # 'a)' uses fig.text because ax_img has non-standard coordinates
    fig.text(0.03, 0.95, 'a)', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
    add_panel_label(ax_p1, 'b)', x=-0.17)
    add_panel_label(ax_p2, 'c)', x=-0.17)

    # ──  data ──────────────────────────────────────────────────────────────
    #region
    data_1 = np.loadtxt(str(_DATA_DIR / 'seq_pi2.csv'), delimiter=',')
    t_raw_1 = data_1[0,:]
    pop_1s0_1 = data_1[1,:]
    pop_3p1_1 = data_1[2,:]
    pop_3p0_1= data_1[3,:]

    data_2 = np.loadtxt(str(_DATA_DIR / 'seq_pi.csv'), delimiter=',')
    t_raw_2 = data_2[0,:]
    pop_1s0_2 = data_2[1,:]
    pop_3p1_2 = data_2[2,:]
    pop_3p0_2= data_2[3,:]
    #endregion

    # ── Method 1 panel ─────────────────────────────────────────────────────────
    ax_p1.set_ylabel('Population', fontsize=FS_LABEL, labelpad=2)
    ax_p1.set_ylim([-0.05, 1.05])
    ax_p1.set_xlim([-0.05, 0.65])
    ax_p1.set_xticks([])
    ax_p1.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)

    ax_p1.scatter(t_raw_1, pop_1s0_1,
                  s=MARKER_S-10, marker='s', ec='k', color=COLOR_1S0, zorder=6)
    ax_p1.scatter(t_raw_1, pop_3p1_1,
                  s=MARKER_S-10, marker='o', ec='k', color=COLOR_3P1, zorder=6)
    ax_p1.scatter(t_raw_1, pop_3p0_1,
                  s=MARKER_S-10, marker='^', ec='k', color=COLOR_3P0, zorder=6)


    ax_p1.plot([], [], marker='s', linestyle='--', label=r'$^1S_0$', color=COLOR_1S0,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    ax_p1.plot([], [], marker='^', linestyle='--', label=r'$^3P_0$', color=COLOR_3P0,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    ax_p1.plot([], [], marker='o', linestyle='--', label=r'$^3P_1$', color=COLOR_3P1,
               markeredgewidth=1, markeredgecolor='black', markersize=7)
    # ax_p1.plot([], [], marker='*', linestyle='--', label=r'$^3P_2$', color=COLOR_3P2,
    #            markeredgewidth=0.8, markeredgecolor='black', markersize=9)

    tpi2 = 0.06
    # ax_p1.axvspan(-.02, tpi2, facecolor='g', alpha=0.2)

    # ax_p1.axvspan(tpi2, 0.64, facecolor='b', alpha=0.2)


    ax_p1.legend(loc='upper right', fontsize=10, frameon=False, ncol=1,labelspacing=0.15,
                 columnspacing=0.5, handlelength=3, handletextpad=0.25)

    # ── Method 2 panel ─────────────────────────────────────────────────────────
    ax_p2.set_ylabel('Population', fontsize=FS_LABEL, labelpad=2)
    ax_p2.set_ylim([-0.05, 1.05])
    ax_p2.set_xlim([-0.05, 0.65])
    ax_p2.tick_params(axis='both', direction='in', which='both', width=TICK_WIDTH)


    ax_p2.scatter(t_raw_2, pop_1s0_2,
                  s=MARKER_S-10, marker='s', ec='k', color=COLOR_1S0, zorder=6)
    ax_p2.scatter(t_raw_2, pop_3p1_2,
                  s=MARKER_S-10, marker='o', ec='k', color=COLOR_3P1, zorder=6)
    ax_p2.scatter(t_raw_2, pop_3p0_2,
                  s=MARKER_S-10, marker='^', ec='k', color=COLOR_3P0, zorder=6)

    ax_p2.set_xlabel(r'Sequence Duration ($\mu$s)', fontsize=FS_LABEL)

    ax_p2.annotate('', xy=(0.62, 0), xytext=(0.62, 1),
                   arrowprops=dict(arrowstyle='<->, head_width=0.5, head_length=1',
                                   color='black', lw=2),
                   zorder=6)
    ax_p2.text(0.6, 0.5, r'94.6%',
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               color='black', zorder=6)

    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig3')
    plt.show()
