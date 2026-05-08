"""
regenerate_all.py — regenerate every figure from scratch using the shared style.

Run from the Figure Code/ directory:
    python regenerate_all.py

Uses a non-interactive backend so no windows are opened.
"""
import matplotlib
matplotlib.use('Agg')   # must be set before any other matplotlib import

import matplotlib.pyplot as plt
from fig_style import save_figure

from figure_3v import make_figure as make_fig_3v
from figure_simulRabi import make_figure as make_fig_simulRabi
from figure_simulline import make_figure as make_fig_simulline
from figure_seqRabi import make_figure as make_fig_seqRabi
from figure_ramsey import make_figure as make_fig_ramsey





figures = [

    ('fig_3v',  make_fig_3v,  {}),
    ('fig_simulRabi',  make_fig_simulRabi,  {}),
    ('fig_simulline',  make_fig_simulline,  {}),
    ('fig_seqRabi',  make_fig_seqRabi,  {}),
    ('fig_ramsey',  make_fig_ramsey,  {})
]

for stem, fn, kwargs in figures:
    print(f"Generating {stem}...", end=' ', flush=True)
    fig = fn(**kwargs)
    save_figure(fig, stem)
    plt.close(fig)
    print("saved.")


print("All figures regenerated.")
