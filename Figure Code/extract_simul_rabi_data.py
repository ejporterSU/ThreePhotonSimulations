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
from scipy.ndimage import gaussian_filter

# fit funcs
#region
def sine(t, A, phi, y0):
    return (A*np.sin(t+phi)+y0)

def exp_sine(t, A, w, phi, tau):
    return (A*np.sin(np.pi*w*t+phi)**2 * np.exp(-t/tau))

#endregion

# extract and process images
_DATA_DIREC = "C:/Users/ggpan/OneDrive - Stanford/Research/manuscripts/DFSequentialPaper/ThreePhotonSimulations/Data"
bins = (1, 120, 180,250)
RID = 75883

save = False
fname = "sim_Rabi1_040726"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,30:280, 50:150] # crop
threshold = 30#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold

ims_gaus = gaussian_filter(ims,5)
# # check bounds
# fig, axs = plt.subplots(1, 3, figsize=(10,10))
# for i in range(3):
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
#     for val in bins:     
#         axs[i].axhline(val, color='red')


# axs[0].imshow(1-ims[0],cmap=plt.get_cmap('Blues')) 
# axs[1].imshow(1-ims_gaus[-7],cmap='gray') 
# axs[1].axis('off')
# axs[2].imshow(1-ims_gaus[-1],cmap=plt.get_cmap('bone'))  
# plt.axis('off')
# plt.savefig(f"{_DATA_DIREC}/cloud_image2.png")
# plt.show()

#%%
plt.figure(figsize=(5, 8))
plt.imshow(1-ims_gaus[-7].T,cmap=plt.get_cmap('bone')) 
plt.axis('off')
# plt.text(0.5, 10, "${}^1S_0$",fontsize=20, color='#1d1c1c', weight='bold')
# plt.text(120.5, 10, "${}^3P_1$",fontsize=20, color='#1d1c1c', weight='bold')
# plt.text(210.5, 10, "${}^3P_0+{}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')

plt.text(0.5, 10, "${}^1S_0+{}^3P_1$",fontsize=20, color='#1d1c1c', weight='bold')
plt.text(120.5, 10, "$\\frac{3}{4} {}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')
plt.text(210.5, 10, "${}^3P_0+\\frac{1}{4} {}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')

#plt.savefig(f"{_DATA_DIREC}/cloud_image1.png")
plt.show()

#np.savetxt(f"{_DATA_DIREC}/Readout2Image.csv", ims[-7], delimiter=",", fmt="%d")


#%%
nrepeats = exp.parameters['nrepeats']
npasses = exp.parameters['npasses']
npoints = exp.parameters['times']['npoints']
times = np.linspace(exp.parameters["times"]['start'], 
                    exp.parameters["times"]['stop'], 
                    npoints)
times = times*1e6
x0 = [np.sum(ims[i,bins[0]:bins[1],:])/np.sum(ims[i,bins[0]:bins[3],:]) for i in range(len(ims))]
x1 = [np.sum(ims[i,bins[1]:bins[2],:])/np.sum(ims[i,bins[0]:bins[3],:]) for i in range(len(ims))]
x2 = [np.sum(ims[i,bins[2]:bins[3],:])/np.sum(ims[i,bins[0]:bins[3],:]) for i in range(len(ims))]


# reshape to match phase
x0 = np.reshape(x0, (npoints, nrepeats) )
x1 = np.reshape(x1, (npoints, nrepeats) )
x2 = np.reshape(x2, (npoints, nrepeats) )
x0_avg = np.mean(x0, axis=1)
x0_std = np.std(x0, axis=1)
x1_avg = np.mean(x1, axis=1)
x1_std = np.std(x1, axis=1)
x2_avg = np.mean(x2, axis=1)
x2_std = np.std(x2, axis=1)

#%%

plt.errorbar(times, x0_avg, yerr=x0_std, c='black', fmt='o')
plt.errorbar(times, x1_avg, yerr=x1_std, c='g', fmt='o')
plt.errorbar(times, x2_avg, yerr=x2_std, c='red', fmt='o')

popt, pcov = curve_fit(exp_sine, times, x2_avg, sigma=x2_std, p0=[1,0.2,0,20], maxfev=20000)

plt.plot(times, exp_sine(times, *popt),c='r')
plt.xlabel('Pulse time (us)')
plt.ylabel('Population')
plt.show()
#%%
np.savetxt(f"{_DATA_DIREC}/Rabi2err.csv", np.array([times,x0_std,x1_std,x2_std]).T, delimiter=",", fmt="%f")

#%%
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
