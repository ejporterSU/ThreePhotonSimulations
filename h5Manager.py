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
        self.df = self.get_h5(RID) # h5 file as a dataframe

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
        
        # grabs a list of all scanned parameters
        self.scans = []
        for key, val in self.parameters.items():
            if isinstance(val, dict):
                self.scans.append(self.dict_to_array(val))

        datasets = list(self.df['datas  ets'].keys())

        if len(self.scans) > 1:
            self.x_vals = np.zeros((len(self.scans[0]), len(self.scans[1]), 2))
            self.y_vals = np.zeros((len(self.scans[0]), len(self.scans[1])))
            for i, val1 in enumerate(self.scans[0]):
                for j, val2 in enumerate(self.scans[1]):
                    self.x_vals[i, j, :] = [val1, val2]
                    self.y_vals[i][j] = self.df[f'datasets/PhaseMeasurement_{i}_{j}'][()]

    


        self.camera_used = (np.char.find(datasets, ".images.") > -1).any()
        self.scan_used = (np.char.find(datasets, "current_scan.") > -1).any()
        self.fit_used = (np.char.find(datasets, ".fits.") > -1).any()

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

        if self.scan_used:
            self.name_space = None
            self.plot_title = None
            self.x = None
            self.x_units = None
            self.x_label = None
            self.x_scale = None
            self.y = None
            self.y_units = None
            self.y_label = None
            self.y_scale = None
            self.extract_plot()

        if self.fit_used:
            self.fit_info = {}
            for ds in datasets:
                if "current_scan.fits." in ds and "guess" not in ds:
                    self.fit_info[ds.split('.')[-1]] = self.df[f'datasets/{ds}'][()]

        self.saved_data = {}
        for ds in datasets:
            if "current_scan" not in ds and "image" not in ds and "plots" not in ds and "fits" not in ds:
                self.saved_data[ds.split('.')[-1]] = self.df[f'datasets/{ds}'][()]

    def help(self):
        print("""Available variables:
                \tdir\n\tpath\n\tdf\n\trid\n\tstart_time\n\trun_time\n\tfile\n\tclass_name
                \tarchive\n\tparameters\n\tscans\n\tbackground_image\n\timages\n\traw_images
                \tfit_info\n\tsaved_data

                Available Methods:
                \tfind_RID\n\tget_h5\n\tdict_to_array\n\t__str__\n\textract_plot\n\tvals_from_images
                """)

    def find_RID(self, RID):
        RID = str(RID)
        for root, dirs, files in os.walk(self.dir):
            for f in files:
                if RID in f and '.csv' not in f:
                    return root + '/' + f

    def get_h5(self, RID):
        path = self.find_RID(RID)
        return h5py.File(path, 'r')

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

    def extract_plot(self):
        try:
            self.name_space = ''
            self.plot_title = self.df['datasets/current_scan.plots.plot_title'][()]
            self.x = self.df['datasets/current_scan.plots.x'][()]
            self.x_units = self.df['datasets/current_scan.plots.x_units'][()]
            self.x_label = self.df['datasets/current_scan.plots.x_label'][()]
            self.x_scale = self.df['datasets/current_scan.plots.x_scale'][()]
            self.y = self.df['datasets/current_scan.plots.y'][()]
            self.y_units = self.df['datasets/current_scan.plots.y_units'][()]
            self.y_label = self.df['datasets/current_scan.plots.y_label'][()]
            self.y_scale = self.df['datasets/current_scan.plots.y_scale'][()]
        except Exception:
            print('No Scan to Generate Plot With')


if __name__ == "__main__":
    RID = 75270
    _DATA_DIREC = "C:/Users/Erik/Desktop/Kasevich Lab/ThreePhotonSimulations\Data"
    exp = ExpViewer(RID, dir=_DATA_DIREC)

