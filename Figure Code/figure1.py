#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path

def load_pdf_as_image(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    return pages[0]  # first page

# --- load your two PDFs ---
img_a = load_pdf_as_image("C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code/energy_diagram.pdf")
img_b = load_pdf_as_image("C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations/Figure Code/three_photon_geom.pdf")

fig, axes = plt.subplots(2, 1, figsize=(6, 10))

for ax, img, label in zip(axes, [img_a, img_b], ["a)", "b)"]):
    ax.imshow(img)
    ax.axis("off")
    offset = 0 if label == 'a)' else 1
    ax.text(
        0 + 0.07 * offset , 1.02, label,
        transform=ax.transAxes, fontweight='bold',
        fontsize=14,
        va="bottom", ha="right"
    )

plt.tight_layout(pad=1.5)
fig.savefig("Figures/fig1.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("Figures/fig1.png", dpi=150, bbox_inches="tight")
plt.show()