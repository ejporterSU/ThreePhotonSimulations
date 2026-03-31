#%%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif', 'axes.linewidth': 0.8})

# --- Style constants ---
FS_LABEL  = 15     # axis labels (left panels)
FS_TICK   = 12      # tick labels (left panels)
FS_CORNER = 9       # in-panel time labels
FS_RLABEL = 12      # axis labels (right panel)
FS_RTICK  = 10      # tick labels (right panel)
LW_MAIN   = 2.5     # solid theory curves
LW_SENS   = 2.0     # dashed sensitivity curves
LW_FIT    = 1.5     # sine / arcsine fits

# --- Colors ---
color_1S0  = "#089EA3"   # ground state
color_3P0  = "#CC0F0F"    # same red
color_3P1  = "#eb45d0"
color_3P2  = "#9fd34b"


fig = plt.figure(figsize=(11, 4.5))

img_width = 0.6                # image occupies 50% of figure width
plot_left = 0.65                # plots start at 55% (leaving a small gap)

# Image placement
ax_img = fig.add_axes([0.04, -0.3, 0.57, 1.6])

# Plot gridspec
gs = GridSpec(2, 1, hspace=0.1,
                    left=plot_left, right=0.96, top=0.93, bottom=0.13)

ax_p1 = fig.add_subplot(gs[0])
ax_p2 = fig.add_subplot(gs[1])





# sequence plot
#region
img = plt.imread('Data/seq_image.png')
ax_img.imshow(img)
ax_img.axis('off')

#endregion

#labels
#region
fig.text(0.01, 0.95, 'a)', fontsize=12, fontweight='bold', va='top', ha='left')
fig.text(0.59, 0.95, 'b)', fontsize=12, fontweight='bold', va='top', ha='left')
fig.text(0.59, 0.52, 'c)', fontsize=12, fontweight='bold', va='top', ha='left')


#endregion

#fake data
#region
t = np.linspace(0, 0.6, 1000)
t_scatter = np.linspace(0, 0.6, 25)
def pop_1s0(ti):
    return np.piecewise(ti, [ti < 0.1, ti >= 0.1], [lambda t: 0.5*(1 + np.cos( t * 0.5*np.pi/0.1)), 0.5])
def pop_3p1(ti):
    return np.piecewise(ti, [ti < 0.1, ti >= 0.1], 
                        [lambda t: 0.5*(1 - np.cos( t * 0.5*np.pi/0.1)), lambda t: 0.25*(1 + np.sin((np.pi/2 + (t-0.1) * np.pi/0.6)))])
def pop_3p0(ti):
    return np.piecewise(ti, [ti < 0.1, ti >= 0.1], [0, lambda t: 0.25*(1 - np.cos(( (t-0.1) * np.pi/0.6)))])
def pop_3p2(ti):
    return np.piecewise(ti, [ti < 0.1, ti >= 0.1], [0, 0])

def pop2_1s0(ti):
    return np.piecewise(ti, [ti < 0.18, ti >= 0.18], 
                        [lambda t: 0.5*(1 + np.cos( t * 0.5*np.pi/0.1)), 
                         0])
def pop2_3p1(ti):
    return np.piecewise(ti, [ti < 0.18, ti >= 0.18], 
                        [lambda t: 0.5*(1 - np.cos( t * np.pi/0.18)), 
                         lambda t: 0.5*(1 + np.cos( (t-0.18) * np.pi/0.4))])
def pop2_3p0(ti):
    return np.piecewise(ti, [ti < 0.18, ti >= 0.18], 
                        [0, 
                         lambda t: 0.5*(1 - np.cos(( (t-0.18) * np.pi/0.4)))])


#endregion


# method 1
#region

ax_p1.set_ylabel('Population', fontsize=FS_RLABEL, labelpad=2)
ax_p1.set_ylim([-0.05, 1.05])
ax_p1.set_xticks([])
ax_p1.tick_params(axis='both', direction='in', which='both', width=1.5)


ax_p1.scatter(t_scatter, pop_1s0(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
            s=50, marker='s', ec='k', color=color_1S0, zorder=6)
ax_p1.scatter(t_scatter, pop_3p1(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
              s=50, marker='^', ec='k',color=color_3P1, zorder=6)
ax_p1.scatter(t_scatter, pop_3p0(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
              s=50, marker='o', ec='k',color=color_3P0, zorder=6)
ax_p1.scatter(t_scatter,  np.random.normal(0, 0.005, len(t_scatter)),
              s=50, marker='*', ec='k',color=color_3P2, zorder=6)

ax_p1.plot(t, pop_1s0(t), color=color_1S0, linestyle='--', linewidth=2)
ax_p1.plot(t, pop_3p1(t), color=color_3P1, linestyle='--', linewidth=2)
ax_p1.plot(t, pop_3p0(t), color=color_3P0, linestyle='--', linewidth=2)
ax_p1.plot(t, pop_3p2(t), color=color_3P2, linestyle='--', linewidth=2)

legend_marker_size = 8
ax_p1.plot([], [], marker='s', linestyle='--', label=r'$^1S_0$', color=color_1S0, markeredgewidth=0.5, markeredgecolor='black', markersize=12)
ax_p1.plot([], [], marker='^', linestyle='--', label=r'$^3P_0$', color=color_3P0, markeredgewidth=0.5, markeredgecolor='black', markersize=12)
ax_p1.plot([], [], marker='o', linestyle='--', label=r'$^3P_1$', color=color_3P1, markeredgewidth=0.5, markeredgecolor='black', markersize=9)
ax_p1.plot([], [], marker='*', linestyle='--', label=r'$^3P_2$', color=color_3P2, markeredgewidth=0.5, markeredgecolor='black', markersize=9)

ax_p1.legend(loc='upper right', fontsize=9, frameon=False, ncol=2,
           columnspacing=0.5, handlelength=3, handletextpad=0.25)



#endregion

# Method 2
#region
ax_p2.set_ylabel('Population', fontsize=FS_RLABEL, labelpad=2)
ax_p2.set_ylim([-0.05, 1.05])
ax_p2.tick_params(axis='both', direction='in', which='both', width=1.5)



ax_p2.plot(t, pop2_1s0(t), color=color_1S0, linestyle='--', linewidth=2)
ax_p2.plot(t, pop2_3p1(t), color=color_3P1, linestyle='--', linewidth=2)
ax_p2.plot(t, pop2_3p0(t), color=color_3P0, linestyle='--', linewidth=2)

ax_p2.scatter(t_scatter, pop2_1s0(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
              s=50, marker='s', ec='k',color=color_1S0, zorder=6)
ax_p2.scatter(t_scatter, pop2_3p1(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
              s=50, marker='^', ec='k',color=color_3P1, zorder=6)
ax_p2.scatter(t_scatter, pop2_3p0(t_scatter) *  np.random.normal(1, 0.05, len(t_scatter)),
              s=50, marker='o', ec='k',color=color_3P0, zorder=6)
# ax_p2.scatter(t_scatter,  np.random.normal(0, 0.005, len(t_scatter)),
#               s=50, marker='*', ec='k',color=color_3P2, zorder=6)
ax_p2.set_xlabel(r'Sequence Duration $T_{Seq}$ ($\mu$s)', fontsize=FS_RLABEL)


ax_p2.annotate('', xy=(0.6, 0), xytext=(0.6, 1),
            arrowprops=dict(arrowstyle='<->, head_width=0.5, head_length=1', color='black', lw=2.5),
            zorder=6)
ax_p2.text(0.58, 0.5, r'98.5(6)%',
        ha='right', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=6)



#endregion



plt.show()
fig.savefig("Figures/fig3.pdf", dpi=300, bbox_inches="tight", facecolor="white")  # save for figure resolution
fig.savefig("Figures/fig3.png", dpi=150, bbox_inches="tight")  # save for previewing
