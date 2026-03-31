#%%
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# ============================================================
# Configuration
# =============================================================
DPI_IMPORT = 600          # rasterization resolution for input PDFs
DPI_EXPORT = 300          # output resolution for saved figure
FIG_WIDTH  = 5.0          # total figure width in inches

PANELS = [
    {
        "path":  "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code/energy_diagram.pdf",
        "label": "a)",
    },
    {
        "path":  "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code/three_photon_geom.pdf",
        "label": "b)",
    },
]

OUTPUT_PDF = "Figures/fig1.pdf"
OUTPUT_PNG = "Figures/fig1.png"

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


# Load all panels
raw_images = [load_pdf_as_array(p["path"], dpi=DPI_IMPORT) for p in PANELS]

# Scale every panel to the same pixel width (use the narrowest to avoid upscaling)
target_w = min(img.shape[1] for img in raw_images)
images = [scale_to_width(img, target_w) for img in raw_images]

# =============================================================
# Build figure with height-proportional rows
# =============================================================

heights_px = [img.shape[0] for img in images]
total_h_px = sum(heights_px)
fig_height = FIG_WIDTH * total_h_px / target_w      # maintain overall aspect ratio

fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
gs = gridspec.GridSpec(
    len(images), 1,
    height_ratios=heights_px,
    hspace=0.1,
)

for i, (img, panel) in enumerate(zip(images, PANELS)):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img, aspect="equal")
    ax.axis("off")

    # Corner label — transAxes puts it in normalized subplot coords
    # so (0.02, 0.98) = top-left corner of every panel, perfectly aligned
    ax.text(
        0.02, 0.98,
        panel["label"],
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )

# =============================================================
# Save
# =============================================================

fig.savefig(OUTPUT_PDF, dpi=DPI_EXPORT, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PNG, dpi=150,        bbox_inches="tight", facecolor="white")
plt.show()