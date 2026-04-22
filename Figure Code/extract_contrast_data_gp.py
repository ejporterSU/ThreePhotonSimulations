#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import heapq
from scipy.optimize import curve_fit
from scipy import stats
# sys.path.insert(0, "C:/Users/sr/Documents/Data Analysis/Python Scripts")  
# sys.path.insert(0, "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations")  
from h5Manager import ExpViewer

# fit funcs
#region
def sine(t, A, phi, y0):
    return (A*np.sin(t+phi)+y0)
def batman(x, n, P0, C):
    return n/np.sqrt(1-((P0-x)/(C/2))**2)

def arcsine_pdf(P, C, offset=0.5):
    x = 2*(P - offset)
    inside = C**2 - x**2
    pdf = np.zeros_like(P)
    mask = inside > 0
    pdf[mask] = 1 / (np.pi * np.sqrt(inside[mask]))
    return pdf

from scipy.ndimage import gaussian_filter1d

def broadened_pdf(P, C, sigma, offset=0.5):
    pdf = arcsine_pdf(P, C, offset)
    dx = P[1] - P[0]
    sigma_bins = sigma / dx
    return gaussian_filter1d(pdf, sigma_bins)

#endregion

# extract and process images
_DATA_DIREC = "C:/Users/ggpan/OneDrive - Stanford/Research/manuscripts/DFSequentialPaper/ThreePhotonSimulations/Data"
bins = (1, 1, 140,260)
RID = 76180

save = False
fname = "phase_contrast_040726"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,40:400, 50:200] # crop
threshold = 20#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold


# check bounds
fig, axs = plt.subplots(1, 3, figsize=(10,10))
for i in range(3):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    for val in bins:     
        axs[i].axhline(val, color='red')
axs[0].imshow(1-ims[0],cmap=plt.get_cmap('Blues')) 
axs[1].imshow(1-ims[-1],cmap=plt.get_cmap('Blues')) 
axs[2].imshow(1-ims[int(len(ims)/2)],cmap=plt.get_cmap('Blues'))  
plt.show()

nrepeats = exp.parameters['nrepeats']
npasses = exp.parameters['npasses']
npoints = exp.parameters["pulse_phase"]['npoints']
phase =2*np.pi * np.linspace(exp.parameters["pulse_phase"]['start'], 
                    exp.parameters["pulse_phase"]['stop'], 
                    npoints)

T_ramsey = exp.parameters["delay"]['start']
fname += f"_{round(T_ramsey*1e6)}us"

xg = [np.sum(ims[i,bins[1]:bins[2],:])/np.sum(ims[i,bins[1]:bins[3],:]) for i in range(len(ims))]
xc = [np.sum(ims[i,bins[2]:bins[3],:])/np.sum(ims[i,bins[1]:bins[3],:]) for i in range(len(ims))]

# reshape to match phase
xg = np.reshape(xg, (npoints, nrepeats) )
xc = np.reshape(xc, (npoints, nrepeats) )
xg_avg = np.mean(xg, axis=1)
xg_std = np.std(xg, axis=1)
xc_avg = np.mean(xc, axis=1)
xc_std = np.std(xc, axis=1)

popt, pcov = curve_fit(sine, phase, xg_avg, sigma=xg_std, p0=[0.5,3.14, 0.5], maxfev=20000)

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(5, 3))

# Plot scatter points and sinusoidal fit on the left
ax1.errorbar(phase, xg_avg, yerr=xg_std, c='black', fmt='o')
ax1.plot(phase, sine(phase, *popt), c='r')
Con = 2*popt[0]
C_err = 2*np.sqrt(np.diag(pcov))[0]

ax1.set_xlabel('Phase (rad.)')
ax1.set_ylabel('1S0 Population')
ax1.set_title(f"Ramsey: RID: {RID}")
ax1.set_ylim(min(xg)-0.05, max(xg)+.05)




# Plot the histogram on the right with the correct orientation
counts, bins, _ = ax2.hist(xg.flatten(), bins=20, alpha=0.5, color='blue', orientation='horizontal', edgecolor='black')
ax2.set_ylim(min(xg)-0.05, max(xg)+.05)
bins = np.array(bins)
bins = (bins[1:] + bins[:-1])/2
popt1,pcov1 = curve_fit(batman,bins, counts, p0=[5,0.5, 0.7], maxfev=100000)
ax2.plot(batman(np.linspace(0,1,100),*popt1),np.linspace(0,1,100), color='red' )

ax2.set_xlim([0, max(counts) + 5])
ax2.set_xlabel('Counts')
ax2.set_yticks([])

ax2.text(0.1, 1, f"Csine = {Con:.3f}+-{C_err:.3f}\nCbat = {popt1[-1]:.3f}+-{np.sqrt(np.diag(pcov1))[0]:.3f}\nP2P: {max(xc.flatten())-min(xc.flatten()):.2f}", 
         transform=ax2.transAxes, va='bottom', ha="left", color='black', fontsize=7)

plt.tight_layout()
plt.show()


pk1 = np.median(heapq.nlargest(15, xg.flatten()))
pk2 = np.median(heapq.nsmallest(15, xg.flatten()))
Con1 = pk1-pk2
C_err1 = np.sqrt(stats.iqr(heapq.nlargest(10, xg.flatten()))**2 + stats.iqr(heapq.nsmallest(10, xg.flatten()))**2) 
#print(f"Peak-to-Peak : {max(xg.flatten())-min(xg.flatten()):.3f}")
print(threshold)
print(f"Peak-to-Peak : {Con1:.3f}\t{C_err1:.3f}")
#print(f"Sine fit     : {Con:.3f}\t{C_err:.3f}")
print(f"Bin width   : {bins[1]-bins[0]:.3f}")
print(counts)
print(bins[-2]-bins[1])
if save:
    data = {
        "phases": phase.tolist(),
        "xc":xc.flatten().tolist(), "xg":xg.flatten().tolist(),
        "xg_avg": xg_avg.tolist(), "xg_std": xg_std.tolist(),
        "xc_avg": xc_avg.tolist(), "xc_std": xc_std.tolist(),
        "hist_bins": bins.tolist(), "hist_counts": counts.tolist(),
        "sine_popt": popt.tolist(),
        "batman_popt": popt1.tolist(),
        "p2p": float(max(xc.flatten()) - min(xc.flatten()))
    }
    with open(f"{_DATA_DIREC}/{fname}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {_DATA_DIREC}/{fname}.json")


# %%
