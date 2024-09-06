# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

#%% Package

import os
import numpy as np

import spyrit.misc.walsh_hadamard as wh
from spas.visualization import plot_reco_without_NN, plot_reco_with_NN 
from spas.metadata import read_metadata, read_metadata_2arms, func_path
from spas.transfer_data_to_girder import transfer_data_2arms, transfer_data
from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction
from spas.reconstruction import reconstruction_hadamard

import time    
import pickle
import csv

#%% INPUT
nb_loop = 1000
delete_old_fig = 0
read_spectral_data = 0
read_had_reco = 0
read_nn_reco = 1

transfer_matched = 0
tranfer = 0
upload_metadata = 0
check_data_exist_in_girder = 0
write_to_csv_file = 0
#%% Begin
t_tot_0 = time.time()
############################ CSV file ##########################
csv_file_path = 'data_in_Girder/data.csv'
csv_exist = os.path.isfile(csv_file_path)
fieldnames = ['setup_version', 'data_folder_name', 'data_name', 'transfered_to_girder', 'had_reco', 'nn_reco', 'check_data_exist_in_girder', 'delete_old_fig']
########################## to be change ############################
setup_version = 'setup_v1.3.1'
# data_folder_name = '2023-03-17_test_Leica_microscope_HCL_Bron'#
# data_folder_name = '2023-04-05_PpIX_at_lab_to_compare_to_hospital'
# data_folder_name = '2023-04-07_PpIX_at_lab_to_compare_to_hospital'
# data_folder_name = '2023-11-21_Arduino_hologram'
# data_folder_name = '2024-02-02_test_chromaticity'
data_folder_name = '2024-02-02_test_reco'

data_file_list = os.listdir('../data/' + data_folder_name)
# data_file_list = ['red_and_black_ink_im_64x64_ti_20ms_zoom_x2']
# data_file_list = ['obj_Arduino_hologram_pos_1_source_White_Zeiss_KL-2500-LCD_lamp_f80mm-P2_Walsh_im_64x64_ti_30ms_zoom_x1']
# data_file_list = ['obj_cat_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1']
data_file_list = ['obj_cat_source_white_LED_f80mm-P2_Walsh_im_64x64_ti_1ms_zoom_x1']
############################ Beginning ##########################
inc = 0
for data_name in data_file_list:
    ########################### path ###################################
    print('data folder : '+data_folder_name)
    print('  -' + data_name)  
    all_path = func_path(data_folder_name, data_name)
    ###################### delete old figures ###########################
    if delete_old_fig == 1:
        fig_list = os.listdir(all_path.overview_path)
        for fig in fig_list:
            if fig.find('HAD_RECO') >=0 or fig.find('NN_RECO') >=0:
                print(fig)
                os.remove(all_path.overview_path + '/' + fig)
    ####################### check if files exist ############
    if os.path.isfile(all_path.had_reco_path):
        exist_had_reco = 1
    else:
        exist_had_reco = 0 
        
    if os.path.isfile(all_path.nn_reco_path):
        exist_nn_reco = 1
    else:
        exist_nn_reco = 0

    if read_nn_reco == 1:
        read_spectral_data = 1
        
        
    # data_overview_list = os.listdir(all_path.overview_path)
    # if len(data_overview_list) > 1 and had_reco == 0:
    #     had_reco = 0
    #     had_reco_for_csv = 1
    # else:
    #     had_reco = 1
    #     had_reco_for_csv = 0
        
    # if os.path.isfile(all_path.nn_reco_path):
    #     nn_reco_for_csv = 1
    # else:
    #     nn_reco_for_csv = 0
    ########################## read spectral data ###########################
    if read_spectral_data == 1:
        print('--- read spectral data')
        file_spectral_data = np.load(all_path.data_path+'_spectraldata.npz')
        try:
            spectral_data = file_spectral_data['spectral_data']
            # print('npz item : spectral_data')
            # for k in file.data_name:
            #     print(k)
        except:
            spectral_data = file_spectral_data['arr_0']
            print('npz item : arr_0')
    ########################### read metadata ###########################
    metadata_path = all_path.data_path + '_metadata.json'
    metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)

    wavelengths = acquisition_parameters.wavelengths
    meas_size = acquisition_parameters.pattern_dimension_x * acquisition_parameters.pattern_dimension_y * 2
    Np = acquisition_parameters.pattern_dimension_x    
    #################### Hadamard reconstruction #######################
    if read_had_reco == 1 and exist_had_reco == 1:
        print('--- read had reco matrix')
        file_had_reco = np.load(all_path.data_path+'_had_reco.npz')  
        GT = file_had_reco['arr_0']
        GT = np.rot90(GT, 2)
        had_reco_for_csv = 1
    elif read_had_reco == 1:
        print('--- reconstruct had reco matrix from spectral data')
        # subsampling
        nsub = 1
        M_sub = spectral_data[:8192//nsub,:]
        patterns_sub = acquisition_parameters.patterns[:8192//nsub]
        
        ### Hadamard Reconstruction
        Q = wh.walsh2_matrix(Np)
        GT = reconstruction_hadamard(patterns_sub, 'walsh', Q, M_sub, Np)
        had_reco_for_csv = 1
    else:
        had_reco_for_csv = 0
    
    if read_had_reco == 1:
        plot_reco_without_NN(acquisition_parameters, GT, all_path)

    ######################### read cam metadata ########################
    try:
        cam_metadata_path = all_path.data_path + '_metadata_cam.pkl'
        
        file = open(cam_metadata_path,'rb')
        cam_metadata = pickle.load(file)
        file.close()
        metadata_cam = 1
    except:
        print('metada of the cam does not exist')
        cam_metadata = []
        metadata_cam = 0
    
    camPar = cam_metadata
    
    #%% Neural Network Reconstruction
    if read_nn_reco == 1:
        print('---------- nn reco ----------')
        t0 = time.time()
        network_param = ReconstructionParameters(
            # Reconstruction network    
            M = round(meas_size/2), #64*64,          # Number of measurements
            img_size = 128,     # Image size
            arch = 'dc-net',    # Main architecture
            denoi = 'unet',     # Image domain denoiser
            subs = 'rect',      # Subsampling scheme
            
            # Training
            data = 'imagenet',  # Training database
            N0 = 10,            # Intensity (max of ph./pixel)
            
            # Optimisation (from train2.py)
            num_epochs = 30,       # Number of training epochs
            learning_rate = 0.001, # Learning Rate
            step_size = 10,        # Scheduler Step Size
            gamma = 0.5,           # Scheduler Decrease Rate   
            batch_size = 256,      # Size of the training batch
            regularization = 1e-7 # Regularisation Parameter
            )
                
        cov_path = 'C:/openspyrit/stat/ILSVRC2012_v10102019/Cov_8_128x128.npy'
        model_folder = 'C:/openspyrit/models/'
        model, device = setup_reconstruction(cov_path, model_folder, network_param)
        plot_reco_with_NN(acquisition_parameters, spectral_data, model, device, network_param, all_path, cov_path)
        print('elapsed time = ' + str(round(time.time()-t0)) + ' s')  
        nn_reco_for_csv = 1                                     
    #%% transfer data to girder
    if tranfer == 1:
        if metadata_cam == 0:
            transfer_data(metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters, 
                                           setup_version, data_folder_name, data_name, upload_metadata)
        else:
            transfer_data_2arms(metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters, camPar,
                                setup_version, data_folder_name, data_name, upload_metadata)
            
            
    inc = inc + 1
    if nb_loop == inc:
        print('stop loop')
        break
    #%% Write dataLog in csv file
    if write_to_csv_file == 1:
        rows = [
            {'setup_version': setup_version,
            'data_folder_name': data_folder_name,
            'data_name': data_name,
            'transfered_to_girder': transfer_matched,
            'had_reco': had_reco_for_csv,
            'nn_reco': nn_reco_for_csv,
            'check_data_exist_in_girder': check_data_exist_in_girder,
            'delete_old_fig': delete_old_fig}
                ]      
        
        with open(csv_file_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if csv_exist == False:
                writer.writeheader()
            writer.writerows(rows)
    
print('total elapsed time = ' + str(round(time.time()-t_tot_0)) + ' s')
# #%%####################### spectra #####################################
# from scipy.signal import savgol_filter

# plot_raw_spectrum = 0
# plot_smooth_spectra = 1
# plot_norm_smooth_spectra = 1


# if data_name == 'Painting_Don_Quichotte_64x64_ti_150ms':
#     y1 = np.mean(np.mean(GT[8:11,29:34,:], axis=1), axis=0) #'white ref'
#     y2 = np.mean(np.mean(GT[25:28,27:30,:], axis=1), axis=0) #'base de la queue'
#     y3 = np.mean(GT[22,33:37,:], axis=0) #'milieu de la queue'
#     y4 = np.mean(np.mean(GT[17:21,19:24,:], axis=1), axis=0) #'background'
# elif data_name == 'Painting_St-Tropez_64x64_ti_100ms':
#     y1 = np.mean(np.mean(GT[19:33,54:57,:], axis=1), axis=0) #'white ref'
#     y2 = np.mean(np.mean(GT[15:19,21:28,:], axis=1), axis=0) #'left house'
#     y3 = np.mean(np.mean(GT[20:22,41:50,:], axis=1), axis=0) #'right house'
#     y4 = np.mean(np.mean(GT[45:56,15:33,:], axis=1), axis=0) #'boat'
    
    
# window_size = 51
# polynomial_order = 4
# ysm1 = savgol_filter(y1, window_size, polynomial_order) 
# ysm2 = savgol_filter(y2, window_size, polynomial_order) 
# ysm3 = savgol_filter(y3, window_size, polynomial_order) 
# ysm4 = savgol_filter(y4, window_size, polynomial_order) 

# if plot_raw_spectrum == 1:
#     plt.figure()
#     plt.plot(wavelengths, y1, color = 'blue')
#     plt.plot(wavelengths, ysm1, color = 'red')
#     plt.title('white ref')
#     plt.grid()
    
#     plt.figure()
#     plt.plot(wavelengths, y2, color = 'blue')
#     plt.plot(wavelengths, ysm2, color = 'red')
#     plt.title('base de la queue')
#     plt.grid()
    
#     plt.figure()
#     plt.plot(wavelengths, y3, color = 'blue')
#     plt.plot(wavelengths, ysm3, color = 'red')
#     plt.title('milieu de la queue')
#     plt.grid()
    
#     plt.figure()
#     plt.plot(wavelengths, y4, color = 'blue')
#     plt.plot(wavelengths, ysm4, color = 'red')
#     plt.title('background')
#     plt.grid()

# if plot_smooth_spectra == 1:
#     plt.figure()
#     plt.plot(wavelengths, ysm1, color = 'green')
#     plt.plot(wavelengths, ysm2, color = 'blue')
#     plt.plot(wavelengths, ysm3, color = 'red')
#     plt.plot(wavelengths, ysm4, color = 'black')
#     plt.legend(['1', '2', '3', '4'])
#     plt.grid()

# if plot_norm_smooth_spectra == 1:
#     cut = 10
#     ym1 = ysm1[cut:-cut]/np.amax(ysm1[cut:-cut])
#     ym2 = ysm2[cut:-cut]/ym1
#     ym3 = ysm3[cut:-cut]/ym1
#     ym4 = ysm4[cut:-cut]/ym1
    
#     plt.figure()
#     plt.plot(wavelengths[cut:-cut], ym2, color = 'blue')
#     plt.plot(wavelengths[cut:-cut], ym3, color = 'red')
#     plt.plot(wavelengths[cut:-cut], ym4, color = 'black')
#     plt.legend(['2', '3', '4'])
#     plt.grid()

# if data_name == 'Falcon_620_WhiteLight_OFF_BlueLaser_ON_im_32x32_Zoom_x1_ti_1000ms#_tc_10.0ms':
#     # 620nm
#     plt.figure()
#     plt.plot(wavelengths, np.mean(np.mean(GT[3:26,5:25,:], axis=1), axis=0))
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('620 nm')
#     plt.grid()

# if data_name == 'Falcon_634_WhiteLight_OFF_BlueLaser_ON_im_64x64_Zoom_x1_ti_50ms#_tc_4.619ms':
#     plt.figure()
#     plt.plot(wavelengths, np.mean(np.mean(GT[15:45,15:45,:], axis=1), axis=0))
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('634 nm')
#     plt.grid()
    
#     plt.figure()
#     plt.plot(wavelengths, GT[32,32,:])
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('634 nm - single pixel')
#     plt.grid()

# if data_name == 'WhiteLight_OFF_BlueLaser_ON_im_64x64_Zoom_x1_ti_100ms#_tc_4.619ms':
#     # 620nm
#     plt.figure()
#     plt.plot(wavelengths, np.mean(np.mean(GT[6:22,19:32,:], axis=1), axis=0))
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('620 nm')
#     plt.grid()
    
#     # 634nm
#     plt.figure()
#     plt.plot(wavelengths, np.mean(np.mean(GT[20:50,37:54,:], axis=1), axis=0))
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('634 nm')
#     plt.grid()
    
#     plt.figure()
#     plt.plot(wavelengths, GT[25,44,:])
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('634 nm - single pixel')
#     plt.grid()
    
#     # S0
#     plt.figure()
#     plt.plot(wavelengths, np.mean(np.mean(GT[27:52,8:30,:], axis=1), axis=0))
#     plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
#     plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
#     plt.title('S_0')
#     plt.grid()
######################### subsampling #######################################
# =============================================================================
# N = 64
# nsub = 2
# M_sub = M[:8192//nsub,:]
# acquisition_parameters.patterns_sub = acquisition_parameters.patterns[:8192//nsub]
# GT_sub = reconstruction_hadamard(acquisition_parameters.patterns_sub, 'walsh', Q, M_sub)
# F_bin_sub, wavelengths_bin, bin_width = spectral_binning(GT_sub.T, acquisition_parameters.wavelengths, 530, 730, 8)
# 
# 
# 
# plot_color(F_bin_sub, wavelengths_bin)
# plt.savefig(fig_had_reco_path + '_wavelength_binning_subsamplig=' + str(nsub) + '.png')
# plt.show()
# =============================================================================

# =============================================================================
# plt.figure
# plt.imshow(GT[:,:,0])
# plt.title(f'lambda = {wavelengths[0]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[0]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,410])
# plt.title(f'lambda = {wavelengths[410]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[410]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,820])
# plt.title(f'lambda = {wavelengths[820]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[820]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,1230])
# plt.title(f'lambda = {wavelengths[1230]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[1230]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(GT[:,:,2047])
# plt.title(f'lambda = {wavelengths[2047]:.2f} nm')
# plt.savefig(fig_had_reco_path + '_' + f'lambda = {wavelengths[2047]:.2f} nm.png')
# plt.show()
# 
# plt.figure
# plt.imshow(np.sum(GT,axis=2))
# plt.title('Sum of all wavelengths')
# plt.savefig(fig_had_reco_path + '_sum_of_wavelengths.png')
# plt.show()
# 
# plt.figure
# plt.scatter(wavelengths, np.mean(np.mean(GT,axis=1),axis=0))
# plt.grid()
# plt.xlabel('Lambda (nm)')
# plt.title('Spectral view in the spatial mean')
# plt.savefig(fig_had_reco_path + '_spectral_axe_of_the_hypercube.png')
# plt.show()
# 
# indx = np.where(GT == np.max(GT))
# sp =  GT[indx[0],indx[1],:]
# plt.figure
# plt.scatter(wavelengths, sp.T)
# plt.grid()
# plt.xlabel('Lambda (nm)')
# plt.title('Spectral view of the max intensity')
# plt.savefig(fig_had_reco_path + '_spectral_axe_of_max_intensity.png')
# plt.show()
# =============================================================================
# #%% Reconstruct with NN

# #%% Setup reconstruction
# network_params = ReconstructionParameters(
#     img_size       = Np,
#     CR             = 1024,
#     denoise        = True,
#     epochs         = 40,
#     learning_rate  = 1e-3,
#     step_size      = 20,
#     gamma          = 0.2,
#     batch_size     = 256,
#     regularization = 1e-7,
#     N0             = 50.0,
#     sig            = 0.0,
#     arch_name      = 'c0mp')

# cov_path   = 'C:/openspyrit/spas/stats/Cov_'+str(Np)+'x'+str(Np)+'.npy'
# mean_path  = 'C:/openspyrit/spas/stats/Average_'+str(Np)+'x'+str(Np)+'.npy'
# model_root = 'C:/openspyrit/spas/models/new-nicolas/'
# H          = wh.walsh2_matrix(Np)/Np
        
# model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
# noise = load_noise('C:/openspyrit/spas/noise-calibration/fit_model2.npz')

# reconstruction_params = {
#     'model'  : model,
#     'device' : device,
#     'batches': 1,
#     'noise'  : noise}

# # network_params = ReconstructionParameters(
# #     img_size=64,
# #     CR=1024,
# #     denoise=True,
# #     epochs=40,
# #     learning_rate=1e-3,
# #     step_size=20,
# #     gamma=0.2,
# #     batch_size=256,
# #     regularization=1e-7,
# #     N0=50.0,
# #     sig=0.0,
# #     arch_name='c0mp',)

# # cov_path = '../stats/new-nicolas/Cov_64x64.npy'
# # mean_path = '../stats/new-nicolas/Average_64x64.npy'
# # H_path = '../stats/new-nicolas/H.npy'
# # model_root = '../models/new-nicolas/'

# # model, device = setup_reconstruction(cov_path, mean_path, H_path, model_root, network_params)
# # noise = load_noise('../noise-calibration/fit_model2.npz')

# # reconstruction_params = {
# #     'model': model,
# #     'device': device,
# #     'batches': 1,
# #     'noise': noise,
# # }

# F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(M.T, acquisition_parameters.wavelengths, 530, 730, 8, 0, noise)
# recon = reconstruct(model, device, F_bin[0:8192//4,:], 1, noise_bin)            
# plot_color(recon, wavelengths_bin)
# plt.savefig(nn_reco_path + '_reco_wavelength_binning.png')
# plt.show()


# #%% transfer data to girder
# transf.transfer_data_to_girder(metadata, acquisition_parameters, spectrometer_params, DMD_params, setup_version, data_folder_name, data_name)
#%% delete plots
# Question = input("Do you want to delete the figures yes [y] ?  ")
# if Question == ("y") or Question == ("yes"):        
#     shutil.rmtree(overview_path)
#     print ("==> figures deleted")










