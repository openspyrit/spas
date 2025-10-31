# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:15:05 2025

@author: admin
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
from spas.reconstruction_SPC1D import hadamard_reco
from spas.acquisition_SPC1D import read_metadata, func_path
from spas.visualization_SCP1D import plot_reco_without_NN

#%% input
plot_fig_without_NN = 1
plot_fig = 1
#%% begin
data_folder_name = '2025-09-08_Compare_V2_vs_V1_SNR'
# data_file_list = os.listdir('../../data/' + data_folder_name)
data_file_list = ['obj_cat_Lc_18.06dB_source_white_LED_Lc_620nm_Gr_1_Walsh_im_128x128_ti_1.0ms_zoom_x2']
# data_name = 'obj_cat_Lc_18.06dB_source_white_LED_Lc_620nm_Gr_2_Walsh_im_64x64_ti_2.0ms_zoom_x2'
# data_name = 'obj_cat_Lc_18.06dB_source_white_LED_Lc_620nm_Gr_1_Walsh_im_128x128_ti_1.0ms_zoom_x2'

for data_name in data_file_list:
    # if data_name.find('655') > 0:
    all_path = func_path(data_folder_name, data_name, ask_overwrite = False)
    ##%% metadata
    output_path = '../../data/' + data_folder_name + '/' + data_name
    saved_DMD_params, saved_spectrograph_params, saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params = read_metadata(output_path + '/metadata.json')
    
    Np= saved_acquisition_params.pattern_dimension_x
    zoom = saved_acquisition_params.zoom
    #%#% reco
    # had_reco = hadamard_reco(data_folder_name, data_name, mean_NA = True, mean_NR = False, save_spectral_data = False, 
    #                          save_spatial_data = False, bin_fact = saved_cam_spec_params.height/Np/zoom, zoom = zoom)
    
    had_reco = hadamard_reco(data_folder_name, data_name, mean_NA = True, mean_NR = False, save_spectral_data = False, 
                             save_spatial_data = False, bin_fact = saved_cam_spec_params.height/Np/1, zoom = zoom)
    ##%% correct had
    had_reco[had_reco < 0] = 0
    if zoom == 1:
        if Np == 128:
            had_reco[81:84,:,:] = np.mean(had_reco)
        elif Np == 64:
            had_reco[40:42,:,:] = np.mean(had_reco)
    if zoom == 2:
        if Np == 128:
            had_reco[65:69,:,:] = 0# np.mean(had_reco)
        elif Np == 64:
            had_reco[33:35,:,:] = 0# np.mean(had_reco)
    #%#% Plot
    if plot_fig_without_NN == 1:
        from matplotlib import pyplot as plt
        
        if len(had_reco.shape) > 3: 
            NLc = had_reco.shape[3]
            for i in range(NLc):
                LLc = saved_acquisition_params.Lc[i][0]    
                plot_reco_without_NN(saved_acquisition_params, had_reco[:,:,:,i], all_path, overwrite = False)
        else:
            plot_reco_without_NN(saved_acquisition_params, had_reco, all_path, overwrite = False)    
    ##%% spectral study
    wavelengths = saved_acquisition_params.wavelengths
    
    init_lambda      = 600#530#
    final_lambda     = 630#550# 
    init_lambda_index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-init_lambda))
    final_lambda_index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-final_lambda))
    
    if Np == 128:
        mean_had = np.mean(had_reco[70:-15,5:-5,init_lambda_index:final_lambda_index], axis=2)
    if Np == 64:
        mean_had = np.mean(had_reco[40:-8,5:-5,init_lambda_index:final_lambda_index], axis=2)
    if Np == 32:
        mean_had = np.mean(had_reco[4:15,5:-5,init_lambda_index:final_lambda_index], axis=2)
        
    moy = np.mean(np.ravel(mean_had))
    std = np.std(np.ravel(mean_had))
    print('------------------------------------------')
    print(data_name)
    print('moy = ' + str(moy))
    print('std = ' + str(std))
    print('------------------------------------------')
    
    
    if plot_fig == 1:
        # mean_had = np.rot90(mean_had, 2)
        plt.figure()
        plt.imshow(np.mean(had_reco[:,:,init_lambda_index:final_lambda_index], axis=2))#(mean_had)
        plt.title('had reco mean L = [' + str(init_lambda) + ' - ' + str(final_lambda) + '] nm')
        plt.colorbar()
        
        plt.figure()
        plt.imshow(mean_had)
        plt.title('cropped mean L = [' + str(init_lambda) + ' - ' + str(final_lambda) + '] nm')
        plt.colorbar()
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