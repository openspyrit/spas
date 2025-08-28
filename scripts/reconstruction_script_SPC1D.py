# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:15:05 2025

@author: admin
"""

from matplotlib import pyplot as plt
import pickle
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
from spas.reconstruction_SPC1D import hadamard_reco
from spas.acquisition_SPC1D import read_metadata, func_path
from spas.visualization_SCP1D import plot_reco_without_NN

data_folder_name = '2025-07-08_tuning'
data_name = 'obj_Cat-Lens-75mm_CL_55mm_source_white_LED_Walsh_im_128x128_ti_1.8ms_zoom_x1'
all_path = func_path(data_folder_name, data_name, ask_overwrite = True)
#%% metadata
output_path = '../../data/' + data_folder_name + '/' + data_name
saved_DMD_params, saved_spectrograph_params, saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params = read_metadata(output_path + '/metadata.json')

Np= saved_acquisition_params.pattern_dimension_x
zoom = saved_acquisition_params.zoom
#%% reco
had_reco = hadamard_reco(data_folder_name, data_name, mean_NA = True, mean_NR = False, save_spectral_data = False, 
                         save_spatial_data = False, bin_fact = saved_cam_spec_params.height/Np/zoom, zoom = zoom)
#%% Plot
from matplotlib import pyplot as plt

if len(had_reco.shape) > 3: 
    NLc = had_reco.shape[3]
    for i in range(NLc):
        LLc = saved_acquisition_params.Lc[i][0]    
        plot_reco_without_NN(saved_acquisition_params, had_reco[:,:,:,i], all_path, overwrite = False)
else:
    plot_reco_without_NN(saved_acquisition_params, had_reco, all_path, overwrite = False)    
#%% plot raw data
plot_fig = True
i = 0
data_path_folder = '../../data/' + data_folder_name + '/' + data_name + '/raw_data/'
data_file_list = os.listdir(data_path_folder)
for file in data_file_list:
    if file.startswith('spectral'):
        data_path = data_path_folder + file
        with open(data_path, "rb") as fp:
            da = pickle.load(fp)
        
        if plot_fig == True:    
            if i <= 5:
                img16 = da
                if file.startswith('spectral'):
                    img8 = img16
                elif file.startswith('spatial'):
                    img8 = (img16/256).astype('uint8')
                plt.figure()
                plt.imshow(img8)
                plt.title(i)
                plt.colorbar()
        
        i = i + 1