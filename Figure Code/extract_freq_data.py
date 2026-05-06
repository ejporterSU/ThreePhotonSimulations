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
bins = (40, 160, 250,350)
#bins = (120, 120, 210,250)

RID = 74794
#RID = 75014

save = False
fname = "sim_Rabi1_040726"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,30:280, 50:110] # crop
threshold = 10#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold

ims_gaus = gaussian_filter(ims,5)
# check bounds
fig, axs = plt.subplots(1, 3, figsize=(10,10))
for i in range(3):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    for val in bins:     
        axs[i].axhline(val, color='red')


axs[0].imshow(1-ims[0],cmap=plt.get_cmap('Blues')) 
axs[1].imshow(1-ims_gaus[-7],cmap='gray') 
axs[1].axis('off')
axs[2].imshow(1-ims_gaus[17],cmap=plt.get_cmap('bone'))  
plt.axis('off')
plt.savefig(f"{_DATA_DIREC}/cloud_image2.png")
plt.show()


#%%
nrepeats = exp.parameters['nrepeats']
npasses = exp.parameters['npasses']
npoints = exp.parameters['frequencies']['npoints']
freqs = np.linspace(exp.parameters["frequencies"]['start'], 
                    exp.parameters["frequencies"]['stop'], 
                    npoints)
freqs = freqs*1e-6
x0 = [np.sum(ims[i,bins[0]:bins[1],:])/np.sum(ims[i,bins[0]:bins[2],:]) for i in range(len(ims))]
x1 = [np.sum(ims[i,bins[1]:bins[2],:])/np.sum(ims[i,bins[0]:bins[2],:]) for i in range(len(ims))]
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

#plt.errorbar(freqs, x0_avg, yerr=x0_std, c='black', fmt='o')
#plt.errorbar(freqs, x1_avg, yerr=x1_std, c='g', fmt='o')
plt.errorbar(freqs*10**6, x1_avg, yerr=x1_std, c='red', fmt='o')
#freqs_mov =(freqs[:-1] + freqs[1:]) / 2
#x2_mov=(x2_avg[:-1] + x2_avg[1:]) / 2

# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# freqs_mov=moving_average(freqs,n=3)*10**6
# x2_mov=moving_average(x2_avg,n=3)

#plt.scatter(freqs_mov,x2_mov)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Population')
plt.show()
#%%
np.savetxt(f"{_DATA_DIREC}/FreqScanNarrow.csv", np.array([freqs*10**3,x2.flatten()]).T, delimiter=",", fmt="%f")



# %%
