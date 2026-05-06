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

def exp_sine(t, A, w, phi, tau,y0):
    return (A*np.sin(2*np.pi*w*t+phi) * np.exp(-t/tau)+y0)

#endregion

# extract and process images
_DATA_DIREC = "C:/Users/ggpan/OneDrive - Stanford/Research/manuscripts/DFSequentialPaper/ThreePhotonSimulations/Data"
bins = (1, 120, 190,250)
RID = 75882

save = True
fname = "Rabi1"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,30:280, 50:110] # crop
threshold = 15#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold

ims_gaus = gaussian_filter(ims,5)
# check bounds
fig, axs = plt.subplots(1, 3, figsize=(10,10))
for i in range(3):
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    for val in bins:     
        axs[i].axhline(val, color='red')

axs[0].imshow(1-ims[192],cmap=plt.get_cmap('Blues')) 
axs[1].imshow(1-ims_gaus[-7],cmap='gray') 
axs[2].imshow(1-ims_gaus[-1],cmap=plt.get_cmap('bone'))  
plt.show()

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

x1[193]=x1[192]
print(np.argmax(x1))
print(x1[160:165])
plt.scatter(np.linspace(0,len(x1),len(x1)),x1)
plt.show()



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

x0_avg=np.delete(x0_avg, 32)
x1_avg=np.delete(x1_avg, 32)
x2_avg=np.delete(x2_avg, 32)
x0_std=np.delete(x0_std, 32)
x1_std=np.delete(x1_std, 32)
x2_std=np.delete(x2_std, 32)
times=np.delete(times, 32)

#%%

#plt.errorbar(times, x0_avg, yerr=x0_std, c='black', fmt='o')
plt.errorbar(times, x1_avg, yerr=x1_std, c='g', fmt='o')
plt.errorbar(times, x2_avg, yerr=x2_std, c='red', fmt='o')

popt, pcov = curve_fit(exp_sine, times, x2_avg, sigma=x2_std, p0=[0.4,0.17,-1.50,10,0.4], maxfev=20000)

plt.plot(times, exp_sine(times, *popt),c='r')
plt.xlabel('Pulse time (us)')
plt.ylabel('Population')
plt.show()
#%%
if save:
    np.savetxt(f"{_DATA_DIREC}/{fname}.csv", np.array([times,x0_avg,x1_avg,x2_avg]).T, delimiter=",", fmt="%f")
    np.savetxt(f"{_DATA_DIREC}/{fname}err.csv", np.array([times,x0_std,x1_std,x2_std]).T, delimiter=",", fmt="%f")


# %%
