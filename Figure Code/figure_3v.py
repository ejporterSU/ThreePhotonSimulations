#%%
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


# .tex files to (re)build before figures run
import subprocess
from pathlib import Path
TEX_FILES = [
    'Figure Code/energy_diagram.tex',
    'Figure Code/doppler_free.tex',
    'Figure Code/three_photon_geom.tex',
]

def compile_tex(tex_rel):
    tex_path = (direc / tex_rel).resolve()

    if not tex_path.is_file():
        raise FileNotFoundError(f"Tex file not found: {tex_path}")
    print(f"Compiling {tex_path.name}...", end=' ', flush=True)
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
    print("done.")





DPI_IMPORT = 600   # rasterization resolution for input PDFs
FIG_WIDTH  = 8.0   # total figure width in inches
FIG_HEIGHT = 5.0
PANELS = [
    {"path": str(_FIGURE_DIR / "three_photon_geom.pdf"),    "label": "a)"},
    {"path": str(_FIGURE_DIR / "energy_diagram.pdf"), "label": "b)"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pdf_as_array(pdf_path, dpi=600):
    """Rasterize first page of a PDF and return as a numpy array."""
    pil_img = convert_from_path(pdf_path, dpi=dpi)[0]
    return np.array(pil_img)


def scale_to_width(img_array, target_width_px):
    """Resize an image array to a target pixel width, preserving aspect ratio."""
    h, w = img_array.shape[:2]
    scale = target_width_px / w
    new_h = int(round(h * scale))
    pil_img = Image.fromarray(img_array)
    return np.array(pil_img.resize((target_width_px, new_h), Image.LANCZOS))


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure():

    for tex in TEX_FILES:
        compile_tex(tex)

    raw_images = [load_pdf_as_array(p["path"], dpi=DPI_IMPORT) for p in PANELS]

    target_w = min(img.shape[1] for img in raw_images)
    images = [scale_to_width(img, target_w) for img in raw_images]

    heights_px = [img.shape[0] for img in images]
    # fig_height = FIG_WIDTH * sum(heights_px) / target_w

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    #gs = gridspec.GridSpec(len(images), 1, height_ratios=heights_px, hspace=0.1)
    gs = gridspec.GridSpec(1,len(images), wspace=0)

    for i, (img, panel) in enumerate(zip(images, PANELS)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img, aspect="equal")
        ax.axis("off")
        add_panel_label(ax, panel["label"], outside=False)


    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_figure(fig, 'fig_3v')
    plt.show()

