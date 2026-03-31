# Figure Style Audit & Consolidation Plan

## Summary

Across `figure1.py`, `figure2.py`, `figure3.py`, and `figure4.py` there are several meaningful style inconsistencies. The issues fall into five categories: color definitions, font size constants, panel label placement/style, global rcParams, and miscellaneous output settings. A `fig_style.py` master style file and a `regenerate_all.py` driver script would resolve all of them.

---

## Identified Discrepancies

### 1. Color Definitions — Major Issue

The 1-photon/3-photon colors are defined with different names *and different hex values* across files:

| Figure | 3-photon color | 1-photon color |
|--------|---------------|----------------|
| fig2   | `color_3v = "#CC0F0F"` | `color_1v = "#0905D3"` |
| fig3   | (not defined — only state colors used) | — |
| fig4   | `c3 = "#C41E1E"` | `c1 = "#1845D4"` |

`#CC0F0F` vs `#C41E1E` are slightly different reds; `#0905D3` vs `#1845D4` are noticeably different blues. Figures that are meant to share a visual language (1-photon vs 3-photon) will therefore use inconsistent colors.

The state-based colors (`color_1S0`, `color_3P0`, `color_3P1`, `color_3P2`) are consistent between fig2 and fig3, but fig4 does not use them at all — it uses `c1`/`c3`/`c1s`/`c3s` for what appear to be the same physical quantities.

### 2. Font Size Constants — Naming & Value Inconsistencies

Each file that defines font size constants uses different names and, in some cases, different values:

| Constant | fig1 | fig2 | fig3 | fig4 |
|----------|------|------|------|------|
| Axis labels | *(none, hardcoded 10)* | `FS_AXIS = 15` | `FS_LABEL = 15` | `FS_LABEL = 15` |
| Panel labels (a, b…) | *(hardcoded 10)* | `FS_PANEL = 12` | *(hardcoded 12)* | *(hardcoded 12)* |
| Right panel labels | — | — | `FS_RLABEL = 12` | `FS_RLABEL = 15` ← **different** |
| Tick labels | — | — | `FS_TICK = 12` | `FS_TICK = 12` |
| Right panel ticks | — | — | `FS_RTICK = 10` | `FS_RTICK = 12` ← **different** |
| Corner time labels | — | — | `FS_CORNER = 9` | `FS_CORNER = 9` |

`FS_RLABEL` and `FS_RTICK` differ between fig3 and fig4 despite being defined with identical names.

### 3. Panel Label Placement & Style — Major Issue

All four figures handle the `a)`, `b)`, `c)` panel labels differently:

| Figure | Method | Position | Font size | Has bbox? |
|--------|--------|----------|-----------|-----------|
| fig1 | `ax.text(0.02, 0.98, …, transform=ax.transAxes)` | Inside axes, top-left | 10 (hardcoded) | Yes — white fill, no edge |
| fig2 | `ax.text(-0.12, 1.02, …, transform=ax.transAxes)` | Outside axes, upper-left | `FS_PANEL = 12` | No |
| fig3 | `fig.text(0.01, 0.95, …)` | Absolute figure coordinates | 12 (hardcoded) | No |
| fig4 | `ax.text(_lx, 1.05, …, transform=ax.transAxes)` | Outside axes, upper-left | 12 (hardcoded) | No |

These four approaches are not interchangeable. `fig.text` with absolute coords (fig3) will break if the layout changes; inside-axes with a bbox (fig1) looks different from the outside-axes approach in figs 2 and 4. The offsets `-0.12` (fig2) vs `-0.14` (fig4) also differ.

### 4. Global rcParams Not Set in fig1

`figure2.py`, `figure3.py`, and `figure4.py` all begin with:

```python
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif', 'axes.linewidth': 0.8})
```

`figure1.py` sets no rcParams at all. Since fig1 embeds rasterized PDFs, the font issue is less acute, but the `axes.linewidth` difference means any future axes added to fig1 would look different.

### 5. Miscellaneous

- **Absolute hardcoded paths in fig1**: The `PANELS` list contains full `C:/Users/Erik/…` Windows paths. All other files use relative paths (e.g., `'Data/seq_image.png'`).
- **`plt.show()` before `savefig`**: fig1 calls `savefig` *then* `show`; figs 2–4 call `show` *then* `savefig`. This is inconsistent and can cause issues in some backends where `show()` resets figure state.
- **`fontweight='bold'` on one ylabel only**: In fig4, the Contrast y-axis label has `fontweight='bold'`; no other figure does this for axis labels.
- **Line width constants**: fig2 defines `LW_THEORY = 2`; fig3 and fig4 define `LW_MAIN = 2.5` for what is conceptually the same type of curve.
- **Marker size approach**: fig2 uses a named constant `MARKER_S = 75` for scatter; figs 3 and 4 hardcode `s=50` or `markersize=4.5`/`7`.

---

## Plan: Master Style File (`fig_style.py`)

Create a single `fig_style.py` in the `Figure Code/` directory. Every figure script imports it with `from fig_style import *` (or `import fig_style as S`). The file should contain:

### Section 1 — rcParams block
Apply the shared `plt.rcParams.update(…)` call once, so importing the module sets the defaults for any figure that uses it.

### Section 2 — Canonical colors
Resolve the hex-value conflicts by picking one authoritative value for each physical quantity. Suggested naming convention:

```
# State colors
COLOR_1S0   = "..."   # ground state
COLOR_3P0   = "..."   # clock state
COLOR_3P1   = "..."
COLOR_3P2   = "..."

# 1-photon vs 3-photon (for linewidth, contrast, etc.)
COLOR_1V    = "..."   # resolved from #0905D3 vs #1845D4
COLOR_3V    = "..."   # resolved from #CC0F0F vs #C41E1E
COLOR_1V_LIGHT = "..."  # lighter shade (was c1s in fig4)
COLOR_3V_LIGHT = "..."  # lighter shade (was c3s in fig4)

COLOR_DLIM  = "#393A3C"   # Doppler limit line
```

### Section 3 — Font sizes (one canonical set)
Pick a single set of names and resolve the fig3 vs fig4 discrepancies:

```
FS_TITLE    = 13    # rcParams base
FS_LABEL    = 15    # all axis labels (resolves FS_AXIS / FS_LABEL / FS_RLABEL conflict)
FS_TICK     = 12    # all tick labels
FS_PANEL    = 12    # a) b) c) labels
FS_CORNER   = 9     # small in-panel annotation labels
FS_LEGEND   = 10    # legend text
```

### Section 4 — Line widths and marker sizes
```
LW_MAIN     = 2.5   # solid theory/model curves
LW_FIT      = 1.5   # sine/arcsine/lorentzian fits
LW_AUX      = 2.0   # dashed auxiliary curves (sensitivity, etc.)
MARKER_S    = 50    # scatter marker size (points²)
MARKER_L    = 7     # marker size for line-plot markers (points)
```

### Section 5 — Panel label helper function
Define a single `add_panel_label(ax, label, loc='outside')` function to standardize placement. This eliminates the four different approaches currently in use. The `'outside'` default positions the label just above and to the left of the axes using `transAxes`, matching the fig2/fig4 style.

### Section 6 — Save helper function
Define a `save_figure(fig, stem)` function that saves both PDF (300 dpi) and PNG (150 dpi) in the `Figures/` directory with consistent `bbox_inches='tight'` and `facecolor='white'`. This also enforces `savefig` being called *before* `plt.show()`.

---

## Plan: Figure Regeneration Script (`regenerate_all.py`)

Create `regenerate_all.py` in `Figure Code/`. It should:

1. Import `fig_style` (which sets rcParams as a side effect).
2. Call each figure module's main generation logic in sequence, or `exec`/`importlib` each file.
3. The cleanest approach is to refactor each `figureN.py` to expose a `make_figure()` function that returns the `fig` object. `regenerate_all.py` then calls each in turn and saves via `fig_style.save_figure(fig, 'figN')`.
4. Print a status line after each figure is saved so the user can see progress.

### Suggested structure of `regenerate_all.py`

```
imports
from fig_style import save_figure

from figure1 import make_figure as make_fig1
from figure2 import make_figure as make_fig2
from figure3 import make_figure as make_fig3
from figure4 import make_figure as make_fig4

for name, fn in [('fig1', make_fig1), ('fig2', make_fig2),
                 ('fig3', make_fig3), ('fig4', make_fig4)]:
    fig = fn()
    save_figure(fig, name)
    print(f"Saved {name}")
```

---

## Migration Checklist (per figure)

When updating each `figureN.py` to use `fig_style`:

- [ ] Remove local color definitions; use `fig_style` constants
- [ ] Remove local font-size constants; use `fig_style` constants
- [ ] Remove `plt.rcParams.update(…)` call (handled by importing `fig_style`)
- [ ] Replace panel label code with `add_panel_label(ax, 'a)')`
- [ ] Replace `fig.savefig(…)` calls with `save_figure(fig, 'figN')`
- [ ] Move all logic into a `make_figure()` function, keeping module-level code to just `from fig_style import *`
- [ ] (fig1 only) Replace absolute Windows paths with relative paths using `pathlib.Path(__file__).parent`
