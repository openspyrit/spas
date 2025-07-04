# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:13:57 2025

@author: admin
"""

import pickle
import numpy as np
from spas.acquisition_SPC1D import read_metadata
from matplotlib import pyplot as plt

#%% read data
data_folder_name = '2025-06-23_wavelength_calib'
data_name = 'obj_nothing_source_HG-1_Oceanoptics_Walsh_im_2x2_ti_500.0ms_zoom_x1'

output_path = '../../data/' + data_folder_name + '/' + data_name
saved_DMD_params, saved_spectrograph_params, saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params = read_metadata(output_path + '/metadata.json')

iNR = 0
iNA = 0
iNp = 0
Nx = saved_cam_spec_params.width
Ny = saved_cam_spec_params.height
Lc = saved_acquisition_params.Lc
NLc = len(Lc)

spectral_data_all = np.empty((Ny, Nx, NLc), dtype = float)
for iLc in range(NLc):
    data_path = output_path + '/raw_data/spectral_NR_' + str(iNR) + '_Gr_' + str(Lc[iLc][1]) + '_Lc_' + str(Lc[iLc][0]) + 'nm_NA_' + str(iNA) + '_NS_' + str(iNp) + '.pkl'
    with open(data_path, "rb") as fp:
        pickle_image = pickle.load(fp)

    spectral_data_all[:, :, iLc] = pickle_image

#%% plot result

for iLc in range(NLc):
    plt.figure()
    plt.imshow(spectral_data_all[:, :, iLc])
    plt.title('Lambda = ' + str(Lc[iLc][0]) + ' - Gr + ' + str(Lc[iLc][1]))

    profile = np.mean(spectral_data_all[290:370, :, iLc], axis = 0)
    
    plt.figure()
    plt.plot(profile)
    plt.title('Lambda = ' + str(Lc[iLc][0]) + ' - Gr + ' + str(Lc[iLc][1]))

#%%post treatment
from scipy.signal import find_peaks

# peaks, di = find_peaks(profile, height = 700)
peaks, properties  = find_peaks(profile, prominence=1, width=20)
