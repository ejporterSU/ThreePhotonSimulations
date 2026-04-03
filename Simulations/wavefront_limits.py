#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

HBAR = const.hbar
H    = const.h
C    = const.c
PI   = np.pi
AMU  = 1.66e-27   # atomic mass unit [kg]
kb   = const.k    # Boltzmann constant [J/K]
MU_B = const.physical_constants['Bohr magneton'][0]  # Bohr magneton [J/T]
EPS0 = const.epsilon_0
eps  = 1e-15  # numerical floor

# ─── Beam parameters ──────────────────────────────────────────────────────────
# (0,1,2) indices -> 689, 688, 679
lambs = np.array([689.4489, 688.020770, 679.288943]) * 1e-9  # [m]
w0s   = np.array([0.5,      0.9,        0.9       ]) * 1e-3  # waist radii [m]
ks    = 2*PI / lambs
Zrs   = PI * w0s**2 / lambs  # Rayleigh ranges [m]
signs = np.array([+1, +1, -1])

print("Rayleigh ranges (m):", Zrs)

# Propagation directions (azimuthal theta, elevation theta_z)
# All beams in the x-y plane (theta_z = 0), so z is transverse to all beams.
theta_0, theta_0z = np.radians(59.6400237748553211),  0.0   # 689 nm
theta_1, theta_1z = np.radians(-59.4380169775801487), 0.0   # 688 nm
theta_2, theta_2z = 0.0,                              0.0   # 679 nm (along +x)

def get_k_hat(theta, theta_z):
    return np.array([np.cos(theta_z)*np.cos(theta),
                     np.cos(theta_z)*np.sin(theta),
                     np.sin(theta_z)])

k_hats = np.array([
    get_k_hat(theta_0, theta_0z),  # 689
    get_k_hat(theta_1, theta_1z),  # 688
    get_k_hat(theta_2, theta_2z),  # 679
])

# Verify plane-wave net k_eff ≈ 0 (angles chosen for this)
k_pw = sum(signs[i] * ks[i] * k_hats[i] for i in range(3))
print(f"Plane-wave net |k_eff| = {np.linalg.norm(k_pw):.4f} rad/m  (target: 0)")

#%%

def gaussian_local_kvec(r, k, Zr, k_hat):
    """
    Local wavevector of a Gaussian beam at a single position.

    The beam propagates along k_hat with waist at the origin.
    Phase: phi(z_b, rho) = k*z_b + k*rho^2 / (2*R(z_b)) - arctan(z_b/Zr)
    where 1/R(z_b) = z_b / (z_b^2 + Zr^2)

    Parameters
    ----------
    r     : (3,) array  – lab-frame position [m]
    k     : float       – wavenumber [rad/m]
    Zr    : float       – Rayleigh range [m]
    k_hat : (3,) array  – unit propagation vector

    Returns
    -------
    (3,) array  – local wavevector [rad/m]
    """
    # Decompose r into axial and transverse beam-frame components
    z_b   = np.dot(r, k_hat)          # axial coordinate
    if abs(z_b) < eps: z_b = eps
    r_par = z_b * k_hat               # parallel component
    r_perp = r - r_par                # transverse vector
    rho   = np.linalg.norm(r_perp)    # transverse distance

    # Transverse unit vector (safe at rho=0)
    rho_hat = r_perp / rho if rho > eps else np.zeros(3)

    # 1/R(z_b) = z_b / (z_b^2 + Zr^2)  — stable everywhere
    denom    = z_b**2 + Zr**2
    Rz   = z_b*(1+(Zr/z_b)**2)

    # d(1/R)/dz = (Zr^2 - z_b^2) / (z_b^2 + Zr^2)^2
    d_inv_Rz = (Zr**2 - z_b**2) / denom**2

    # Gouy phase: d/dz[-arctan(z/Zr)] = -Zr/(z_b^2+Zr^2)
    gouy_kz  = -Zr / denom

    # Axial and transverse components of local k
    k_axial = k + (k * rho**2 / 2) * d_inv_Rz + gouy_kz
    k_trans = k * rho / Rz

    return k_axial * k_hat + k_trans * rho_hat


def residual_kvec(r):
    """k_eff(r) = k_689(r) + k_688(r) - k_679(r)"""
    return sum(signs[i] * gaussian_local_kvec(r, ks[i], Zrs[i], k_hats[i])
               for i in range(3))


# ─── Compute heatmaps at three z-slices ───────────────────────────────────────
# Subtract k_eff at origin so heatmaps show the Gaussian wavefront variation.

k0_vec = residual_kvec(np.zeros(3))
print(f"k_eff at origin: {k0_vec}  |k0| = {np.linalg.norm(k0_vec):.4f} rad/m")

N         = 350
extent_um = 150  # µm half-width of plot window
x_vals    = np.linspace(-extent_um * 1e-6, extent_um * 1e-6, N)
y_vals    = np.linspace(-extent_um * 1e-6, extent_um * 1e-6, N)

z_slices     = [0.0, 100e-6, -100e-6]
slice_labels = ['z = 0', 'z = +100 µm', 'z = -100 µm']

def delta_kvec_slice(z_val, x_vals, y_vals):
    """
    |k_eff(r) - k_eff(0)| on a 2-D x-y grid at fixed lab z.
    Returns (Ny, Nx) magnitude array.
    """
    mag = np.empty((len(y_vals), len(x_vals)))
    mag_abs = np.empty((len(y_vals), len(x_vals)))
    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            r = np.array([x, y, z_val])
            delta = residual_kvec(r) - k0_vec
            mag[j, i] = np.linalg.norm(delta)
            mag_abs[j, i] = np.linalg.norm(residual_kvec(r))

    return mag, mag_abs


#%%
mags = []
mags_abs = []
for z in z_slices:
    m, m_abs = delta_kvec_slice(z, x_vals, y_vals)

    mags.append(m)
    mags_abs.append(m_abs)

    print("Relative")
    print(f"  z={z*1e6:+.0f} um  |dk| min={m.min():.6f}  max={m.max():.6f}  "
          f"center={m[N//2, N//2]:.6f}  rad/m")
    print("Absolute")
    print(f"  z={z*1e6:+.0f} um  |dk| min={m_abs.min():.6f}  max={m_abs.max():.6f}  "
          f"center={m_abs[N//2, N//2]:.6f}  rad/m")

# ─── Plot ─────────────────────────────────────────────────────────────────────
#%%
cloud_sigma_xy = 43e-6
theta_circ     = np.linspace(0, 2*PI, 400)
sigma_um       = cloud_sigma_xy * 1e6
extent_plot    = [-extent_um, extent_um, -extent_um, extent_um]

vmax_rel = max(m.max() for m in mags)
vmax_abs = max(m.max() for m in mags_abs)
vmin_abs = min(m.min() for m in mags_abs)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

row_data   = [mags,     mags_abs]
row_vmins  = [0,        vmin_abs]
row_vmaxs  = [vmax_rel, vmax_abs]
row_labels = [r'$|\vec{k}_{\rm eff}(\mathbf{r}) - \vec{k}_{\rm eff}(\mathbf{0})|$  (rad/m)',
              r'$|\vec{k}_{\rm eff}(\mathbf{r})|$  (rad/m)']
row_titles = ['Relative (Gaussian variation)', 'Absolute']

for row, (data, vmin, vmax, cbar_label, row_title) in enumerate(
        zip(row_data, row_vmins, row_vmaxs, row_labels, row_titles)):

    for col, (ax, mag, slice_label) in enumerate(zip(axes[row], data, slice_labels)):
        im = ax.imshow(mag, extent=extent_plot, origin='lower',
                       cmap='inferno', vmin=vmin, vmax=vmax, aspect='equal')

        ax.plot(sigma_um * np.cos(theta_circ),
                sigma_um * np.sin(theta_circ),
                'w--', lw=1.2, alpha=0.8, label=f'1-sigma ({sigma_um:.0f} um)')

        for kh, c, name in zip(k_hats,
                                ['deepskyblue', 'lime', 'orange'],
                                ['689 nm', '688 nm', '679 nm']):
            ax.annotate('', xy=(kh[0]*80, kh[1]*80), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
            ax.text(kh[0]*90, kh[1]*90, name, color=c, fontsize=7,
                    ha='center', va='center')

        ax.set_title(slice_label, fontsize=11)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')

    axes[row, 0].legend(loc='upper right', fontsize=7, framealpha=0.4)
    axes[row, 0].set_ylabel(f'{row_title}\ny (um)', fontsize=9)
    fig.colorbar(im, ax=axes[row].tolist(), label=cbar_label, shrink=0.85, pad=0.01)

fig.suptitle(
    r'$\vec{k}_{\rm eff}(\mathbf{r}) = \vec{k}_{689}(\mathbf{r}) + \vec{k}_{688}(\mathbf{r}) - \vec{k}_{679}(\mathbf{r})$'
    '  — Gaussian beam local wavevectors\n'
    r'$w_0$ = 0.5 mm (689 nm),  0.9 mm (688, 679 nm);  waists at cloud centre',
    fontsize=11
)

plt.show()
