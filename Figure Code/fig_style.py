"""
fig_style.py — shared matplotlib style for all ThreePhotonSimulations figures.

Import at the top of every figure file:
    from fig_style import *
Importing this module automatically applies the shared rcParams.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# ── Global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
})

# ── Colors: atomic states ──────────────────────────────────────────────────────
#COLOR_1S0 = "#089EA3"   # ground state
#COLOR_3P0 = "#CC0F0F"   # clock state  (also used as COLOR_3V)
#COLOR_3P1 = "#eb45d0"
#COLOR_3P2 = "#9fd34b"

COLOR_1S0 = "#5D2BF5"   # ground state
COLOR_3P0 = "#CC0F0F"   # clock state  (also used as COLOR_3V)
COLOR_3P1 = "#FF6B35"
COLOR_3P2 = "#167E29"

# ── Colors: 1-photon vs 3-photon comparison ────────────────────────────────────
COLOR_1V       = "#1845D4"   # 1-photon solid
COLOR_3V       = "#CC0F0F"   # 3-photon solid  (= COLOR_3P0)
COLOR_1V_LIGHT = "#90AAFF"   # 1-photon light / auxiliary dashed
COLOR_3V_LIGHT = "#FF9090"   # 3-photon light / auxiliary dashed
COLOR_DLIM     = "#393A3C"   # Doppler limit reference line

# ── Font sizes ─────────────────────────────────────────────────────────────────
FS_LABEL  = 15   # axis labels
FS_TICK   = 12   # tick labels
FS_PANEL  = 12   # a) b) c) panel corner labels
FS_CORNER = 9    # small in-panel annotations (time stamps, etc.)
FS_LEGEND = 10   # legend text

# ── Line widths ────────────────────────────────────────────────────────────────
LW_MAIN = 2.5   # primary theory / model curves
LW_AUX  = 2.0   # secondary / dashed auxiliary curves
LW_FIT  = 1.5   # data fits (Lorentzian, Gaussian, sine, etc.)

# ── Marker sizes ───────────────────────────────────────────────────────────────
MARKER_S = 50   # scatter marker area (pt²) — main data points
MARKER_L = 7    # marker radius (pt) — markers on line plots

# ── Tick style ─────────────────────────────────────────────────────────────────
TICK_WIDTH = 1.5

# ── Output ─────────────────────────────────────────────────────────────────────
_FIGURES_DIR = Path(__file__).parent.parent / "Figures"


def save_figure(fig, stem):
    """Save *fig* as PDF (300 dpi) and PNG (150 dpi) in the Figures/ directory."""
    _FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(_FIGURES_DIR / f"{stem}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(_FIGURES_DIR / f"{stem}.png", dpi=150, bbox_inches='tight', facecolor='white')


def add_panel_label(ax, label, outside=True, x=None, y=None):
    """Add a bold panel label (e.g. 'a)') to an axes corner.

    outside=True  (default): just above/left of the axes, no background box.
    outside=False           : inside top-left corner, with a white background box.
    x, y override the default position in axes-fraction coordinates.
    """
    if outside:
        _x = -0.13 if x is None else x
        _y =  1.05 if y is None else y
        ax.text(_x, _y, label, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
    else:
        _x = 0.02 if x is None else x
        _y = 0.98 if y is None else y
        ax.text(_x, _y, label, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
