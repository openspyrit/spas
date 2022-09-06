# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
#from spyrit.learning.model_Had_DCAN import Hadamard_Transform_Matrix
import spyrit.misc.walsh_hadamard as wh

from spas import *
#from singlepixel import read_metadata, reconstruction_hadamard
#from singlepixel import *
import spas.transfer_data_to_girder as transf

#import singlepixel.transfer_data_to_girder as transf
import os
import shutil
import scipy
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

# Matlab patterns
#file = loadmat('../data/matlab.mat')
#Q = file['Q']

# fht patterns
#Q = Hadamard_Transform_Matrix(64)

#%% Reconstruction without NN
zoom = '1'
group = '3'
########################## to be change ############################
setup_version = 'setup_v1.3'#'setup_v1.2'
data_folder_name = '2022-01-25_spatial_resolution_USAF'#'2021-10-22_magnification_usaf_WhiteLED'
#data_name_list = ['zoom_x'+zoom+'_usaf_group'+group]
data_name_list = ['USAFx'+zoom]
ii = 0
for data_name in data_name_list:
    ii = ii + 1
    print('--------' + data_name + '--------')
    ########################### path ###################################
    #data_path = 'C:/spas/Programs/Python/data/' + data_folder_name + '/' + data_name + '/' + data_name
    data_path = '../data/' + data_folder_name + '/' + data_name + '/' + data_name
    had_reco_path = data_path + '_had_reco.npz'
    reco_path = '../data/' + data_folder_name + '/result/'
    ########################## read raw data ###########################
    file = np.load(data_path+'_spectraldata.npz')
    M = file['spectral_data']#['arr_0']#['spectral_data']
    Q = wh.walsh2_matrix(64)
    
    metadata_path = data_path + '_metadata.json'
    metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)
    
    GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, M)
    
    metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
    wavelengths = acquisition_metadata.wavelengths
    #F_bin, wavelengths_bin, bin_width = spectral_binning(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)
    
    
    plt.figure
    plt.imshow(np.sum(GT, axis=2))
    plt.colorbar();
    plt.show() 
    
    spectrum = np.mean(np.mean(GT,axis=1),axis=0)
    
    plt.figure
    plt.plot(wavelengths, spectrum)
    plt.show()
    
    wavelength1 = 580# 550#590
    wavelength2 = 600# 590#610
    indx1_vec = abs(wavelengths - wavelength1)
    indx1 = np.argmin(indx1_vec)
    indx2_vec = abs(wavelengths - wavelength2)
    indx2 = np.argmin(indx2_vec)
    print('wavelength1 = '+str(wavelengths[indx1]))
    print('wavelength2 = '+str(wavelengths[indx2]))
    
    GT2 = GT[:,:,indx1:indx2]
    
    plt.figure
    plt.imshow(np.sum(GT2, axis=2))#, cmap='gray')
    #plt.colorbar();
    plt.axis('off')
    plt.savefig(reco_path +'zoom_x'+zoom+ '_reco_wavelength_'+str(wavelength1)+'-'+str(wavelength2)+'nm_gray.png', bbox_inches='tight',pad_inches = 0)
    plt.show() 
    
    zoom_tab = [1, 2, 3, 4, 6, 12]
    res_vert_tab = [2.83, 6.35, 8.98, 12.7, 17.96, 40.3]
    res_horz_tab = [3.17, 5.66, 8.98, 12.7, 17.96, 40.3]
    res_best_tab = [3.17, 6.35, 8.98, 12.7, 17.96, 40.3]
    res_worst_tab = [2.83, 5.66, 8.98, 12.7, 17.96, 35.9]
    second_acq = [2.83, 5.66, 8.98, 12.7, 17.96, 40.3]
    #[G1-E4 , G2-E4 , G3-E2 , G3-E5 , G4-E2 , G5-E3]
    x_th = np.array([0, 1, 2, 3, 4, 6, 12])
    y_th = np.round(x_th/(2*13.67e-3*12*0.9)*100)/100#(2*13.67e-3*12)#
    
    
    plt.figure
    plt.plot(zoom_tab, second_acq, '*', x_th, y_th)
    plt.grid
    plt.show() 
    
# #%% Reconstruct with NN
# network_params = ReconstructionParameters(
#     img_size=64,
#     CR=1024,
#     denoise=True,
#     epochs=40,
#     learning_rate=1e-3,
#     step_size=20,
#     gamma=0.2,
#     batch_size=256,
#     regularization=1e-7,
#     N0=50.0,
#     sig=0.0,
#     arch_name='c0mp',)

# cov_path = 'C:/openspyrit/spas/stats/new-nicolas/Cov_64x64.npy'
# mean_path = 'C:/openspyrit/spas/stats/new-nicolas/Average_64x64.npy'
# model_root = 'C:/openspyrit/spas/models/new-nicolas/'
# H = wh.walsh2_matrix(64)/64
        
# model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
# noise = load_noise('C:/openspyrit/spas/noise-calibration/fit_model2.npz')

# reconstruction_params = {
#     'model': model,
#     'device': device,
#     'batches': 1,
#     'noise': noise,
# }

# F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(M.T, acquisition_parameters.wavelengths, 530, 730, 8, noise)
# recon = reconstruct(model, device, F_bin[:,0:8192//4], 1, noise_bin)            
# plot_color(recon, wavelengths_bin)
# #plt.savefig(nn_reco_path + '_reco_wavelength_binning.png')
# plt.show()

# #%% transfer data to girder
# transf.transfer_data_to_girder(metadata, acquisition_parameters, spectrometer_params, DMD_params, setup_version, data_folder_name, data_name)
# #%% delete plots
# Question = input("Do you want to delete the figures yes [y] ?  ")
# if Question == ("y") or Question == ("yes"):        
#     shutil.rmtree(overview_path)
#     print ("==> figures deleted")










