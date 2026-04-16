#%%

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json


class ExpViewer:
    def __init__(self, RID, dir=None):
        self.dir = dir if dir is not None else os.getcwd() # sets internal directory
        self.path = self.find_RID(RID) # attempts to find path
        print(f"found RID at: {self.path}")
        self.df = self.get_h5() # h5 file as a dataframe


        # extract surface info
        self.rid = self.df['rid'][()]
        self.artiq_version = self.df['artiq_version'][()]
        self.start_time = self.df['start_time'][()]
        self.run_time = self.df['run_time'][()]

        # expid contains arguemnts, filename, classs, etc 
        self.expid = json.loads(self.df['expid'][()])
        self.file = self.expid['file']
        self.class_name = self.expid['class_name']
        self.parameters = self.expid['arguments']

        # archive contains persistent variables
        archive_vars = list(self.df['archive'].keys())
        archive_vals = [self.df[f'archive/{var}'][()] for var in archive_vars]
        self.archive = dict(zip(archive_vars, archive_vals))
 
   
        datasets = list(self.df['datasets'].keys())

        self.camera_used = (np.char.find(datasets, ".images.") > -1).any()
        self.scan_used = (np.char.find(datasets, "current_scan.") > -1).any()
        self.fit_used = (np.char.find(datasets, ".fits.") > -1).any()

        if self.scan_used:
            is_2d = 'current_scan.plots.dim1.x' in datasets
            if is_2d:
                self.x1 = self.df['datasets/current_scan.plots.dim1.x'][()]
                self.x0 = self.df['datasets/current_scan.plots.x'][()]
            else:
                self.x = self.df['datasets/current_scan.plots.x'][()]

        # extracts all images
        if self.camera_used:
            self.raw_images = []
            self.images = []
            ind = 0
            self.background = np.array(self.df['datasets/detection.images.background_image'][()])
            while True:
                if f'detection.images.{ind}' not in datasets:
                    self.N = ind
                    break
                self.images.append(np.array(self.df[f'datasets/detection.images.{ind}'][()]))
                try:
                    self.raw_images.append(np.array(self.df[f'datasets/detection.images.Raw_{ind}'][()]))
                except Exception:
                    pass
                ind += 1


        skip_words = ['image']
        self.data = {}
        for ds in datasets:
            if not any(word in ds for word in skip_words):
                self.data[ds] = self.df[f'datasets/{ds}'][()]


    def find_RID(self, RID):
        RID = str(RID)
        for root, dirs, files in os.walk(self.dir):
            for f in files:
                if RID in f and '.csv' not in f:
                    return root + '/' + f
        raise Exception(f"No RID ({RID}) found...")

    def get_h5(self):
        return h5py.File(self.path, 'r')

    def close(self):
        self.df.close()

    def dict_to_array(self, myDict):
        """
        turns a dictionary defining a scan range into a numpy array
        """
        return np.linspace(myDict['start'], myDict['stop'], myDict['npoints'])

    def __str__(self):
        """
        method for printing exp info with print()
        """
        res = "EXPERIMENT\n-------------\n"
        res += f'class: {self.class_name}\n'
        res += f'file: {self.file}\n'
        res += f'artiq_version: {self.artiq_version}\n'
        res += f'rid: {self.rid}\n'
        res += f'run_time: {self.run_time}\n'
        res += f'start_time: {self.start_time}\n\n'

        res += 'PARAMETERS\n-----------\n'
        for key, val in self.parameters.items():
            res += f'{key}: {val}\n'
        for key, val in self.archive.items():
            res += f'{key}: {val}\n'

        return res

if __name__ == "__main__":
    RID = 75270 # 2D
    # RID = 75205 # 1D
    _DATA_DIREC = "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations\Data"
    exp = ExpViewer(RID, dir=_DATA_DIREC)



