import os
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import json



class ExpViewer:
    def __init__(self, RID, dir=None):
        self.dir = dir if dir is not None else os.getcwd()
        self.path = self.find_RID(RID)
        self.df = self.get_h5(RID)
        self.rid = self.df['rid'][()]
        self.artiq_version = self.df['artiq_version'][()]
        self.start_time = self.df['start_time'][()]
        self.run_time = self.df['run_time'][()]

        self.expid = json.loads(self.df['expid'][()])
        self.file = self.expid['file']
        self.class_name = self.expid['class_name']

        archive_vars = list(self.df['archive'].keys())
        archive_vals = [self.df[f'archive/{var}'][()] for var in archive_vars]
        self.archive = dict(zip(archive_vars, archive_vals))
        self.parameters = self.expid['arguments']

        self.scans = {}
        for key, val in self.parameters.items():
            if type(val) == dict:
                self.scans[key] = self.dict_to_array(val)

        self.custom_vals = {}

        datasets = list(self.df['datasets'].keys())
        self.camera_used = (np.char.find(datasets, ".images.")>-1).any()
        self.scan_used = (np.char.find(datasets, "current_scan.")>-1).any()
        self.fit_used = (np.char.find(datasets, ".fits.")>-1).any()

        if self.camera_used:
            self.raw_images = []
            self.images = []
            ind = 0
            self.background = np.array(self.df[f'datasets/detection.images.background_image'][()])
            while True:
                if f'detection.images.{ind}' not in datasets:
                    self.N = ind
                    break
                self.images.append(np.array(self.df[f'datasets/detection.images.{ind}'][()]))
                try:
                    self.raw_images.append(np.array(self.df[f'datasets/detection.images.Raw_{ind}'][()]))
                except:
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
        ## extracting additional
        self.saved_data = {}
        for ds in datasets:
            if "current_scan" not in ds and "image" not in ds and "plots" not in ds and "fits" not in ds:
                self.saved_data[ds.split('.')[-1]] = self.df[f'datasets/{ds}'][()]




    def help(self):
        print("""
Available variables:\n\tdir\n\tpath\n\tdf\n\trid\n\tstart_time\n\trun_time\n\tfile\n\tclass_name\n\tarchive\n\tparameters\n\tscans\n\tbackground_image\n\timages\n\traw_images\n\tfit_info\n\tsaved_data

Available Methods:\n\tfind_RID\n\tget_h5\n\tdict_to_array\n\t__str__\n\textract_plot\n\tgenerate_plot\n\tvals_from_images

        """)


    def find_RID(self, RID):
        RID = str(RID)
        for root, dirs, files in os.walk(self.dir):
            for f in files:
                if RID in f and '.csv' not in f:
                    return root + '/' + f

    def get_h5(self, RID):
        path = self.find_RID(RID)
        df = h5py.File(path, 'r')
        return df

    def dict_to_array(self, myDict):
        return np.linspace(myDict['start'], myDict['stop'], myDict['npoints'])

    def __str__(self):
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
        except:
            print('No Scan to Generate Plot With')
            pass


    def generate_plot(self):

        fig, axe = plt.subplots(figsize=(10, 6))
        axe.scatter(self.x/self.x_scale, self.y, c='r')
        axe.set_xlabel(f"{self.x_label} ({self.x_units})")
        axe.set_ylabel(f"{self.y_label} ({self.y_units})")
        axe.set_title(self.plot_title)

        x_min, y_min = min(self.x)/self.x_scale, min(self.y)
        x_max, y_max = max(self.x)/self.x_scale, max(self.y)
        del_x = (x_max-x_min)*0.1
        del_y = (y_max-y_min)*0.1
        axe.set_xlim(x_min-del_x, x_max+del_x)
        axe.set_ylim(y_min-del_y, y_max+del_y)
        axe.grid()
        if self.fit_used:
            plt.plot(self.x/self.x_scale, self.fit_info["fitline"], c='r')
        plt.show()

    def vals_from_images(self, myFunc, name, **kwargs):
        res = np.zeros(self.N)
        for i in range(self.N):
            res[i] = myFunc(self.images[i], **kwargs)
        self.custom_vals[name] = res
