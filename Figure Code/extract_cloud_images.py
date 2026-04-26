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
RID = 75882

save = False
fname = "sim_Rabi1_040726"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,30:280, 50:150] # crop
threshold = 30#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold

ims_gaus = gaussian_filter(ims,5)[-1]

RID = 75883

save = False
fname = "sim_Rabi1_040726"

exp = ExpViewer(RID, dir=_DATA_DIREC)
ims = np.array(exp.images)
ims = ims[:,10:280, 50:150] # crop
threshold = 30#max(ims[1:20].flatten())/50
ims = np.where(ims > threshold, ims, 0) #threshold

ims_gaus2 = gaussian_filter(ims,5)[-41]

plt.imshow(1-ims_gaus2.T,cmap=plt.get_cmap('bone')) 
#%%
fig, axs = plt.subplots(2, 1, figsize=(5,5))
axs[0].imshow(1-ims_gaus.T,cmap=plt.get_cmap('bone')) 
axs[0].axis('off')
axs[0].text(0.5, 10, "${}^1S_0$",fontsize=20, color='#1d1c1c', weight='bold')
axs[0].text(120.5, 10, "${}^3P_1$",fontsize=20, color='#1d1c1c', weight='bold')
axs[0].text(210.5, 10, "${}^3P_0+{}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')
axs[0].text(150, 100, f'Readout 1', ha='center', va='bottom',
                    fontsize=20, color='#1d1c1c')

axs[1].imshow(1-ims_gaus2.T,cmap=plt.get_cmap('bone')) 
axs[1].axis('off')
axs[1].text(0.5, 10, "${}^1S_0+{}^3P_1$",fontsize=20, color='#1d1c1c', weight='bold')
axs[1].text(130.5, 10, "$\\frac{3}{4} {}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')
axs[1].text(230.5, 10, "${}^3P_0+\\frac{1}{4} {}^3P_2$",fontsize=20, color='#1d1c1c', weight='bold')
axs[1].text(160, 100, f'Readout 2', ha='center', va='bottom',
                    fontsize=20, color='#1d1c1c')
#%%
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
# %%
