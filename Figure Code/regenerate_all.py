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

from figure1 import make_figure as make_fig1
from figure2 import make_figure as make_fig2
from figure3 import make_figure as make_fig3
from figure4 import make_figure as make_fig4, VERTICAL

figures = [
    ('fig1',                     make_fig1,  {}),
    ('fig2',                     make_fig2,  {}),
    ('fig3',                     make_fig3,  {}),
    (f'fig4{"v" if VERTICAL else "h"}', make_fig4, {'vertical': VERTICAL}),
]

for stem, fn, kwargs in figures:
    print(f"Generating {stem}...", end=' ', flush=True)
    fig = fn(**kwargs)
    save_figure(fig, stem)
    plt.close(fig)
    print("saved.")


print("All figures regenerated.")
