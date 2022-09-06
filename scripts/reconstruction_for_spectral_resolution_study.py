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
########################## to be change ############################
setup_version = 'setup_v1.2'
data_folder_name = '2021-12-15_spectral_resolution_HgAr_Lamp'#'2021-07-26-spectral-analysis'
data_name_list = ['spot_at_the_bottom_right', 'spot_at_the_center', 'spot_at_the_top_left']
ii = 0

for data_name in data_name_list:
    ii = ii + 1
    print('--------' + data_name + '--------')
    ########################### path ###################################
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
    
    if ii == 1:
        GT_1 = np.sum(GT, axis=2)
    elif ii == 2:
        GT_2 = np.sum(GT, axis=2)
    elif ii == 3:
        GT_3 = np.sum(GT, axis=2)
    
    plt.figure(ii)
    plt.imshow(np.sum(GT, axis=2))
    plt.colorbar();
    plt.grid()
    plt.show() 
    
    # GT2[GT2<10000] = 0
    
    # plt.figure
    # plt.imshow(GT2)
    # plt.colorbar();
    # plt.show() 
    # plt.title('denoised');
    
    #spectrum = np.mean(np.mean(GT,axis=1),axis=0)
    if ii == 1:
        spectrum = np.mean(np.mean(GT[10:20,8:18,:],axis=1),axis=0)
        thershold_peak = 2
        delete_peak = 2
    elif ii == 2:
        spectrum = np.mean(np.mean(GT[30:40,30:40,:],axis=1),axis=0)
        thershold_peak = 2
        delete_peak = 3
    else:
        spectrum = np.mean(np.mean(GT[50:60,45:55,:],axis=1),axis=0)
        thershold_peak = 1
        delete_peak = 2
    
    if ii == 1:
        spectrum_1 = spectrum
    elif ii == 2:
        spectrum_2 = spectrum
    elif ii == 3:
        spectrum_3 = spectrum
    
    plt.figure(ii)
    plt.plot(wavelengths, spectrum/np.amax(spectrum))
    plt.grid()
    plt.xlabel('Lambda (nm)')
    plt.ylabel('Intensity')
    #plt.title('Spectral view in the spatial mean')
    plt.savefig(reco_path + 'spectral_resolution_spectrum_'+str(ii)+'.pdf', bbox_inches='tight',pad_inches = 0)
    plt.show
    
    
    # 1D interpolation
    wavelength_length = len(wavelengths)
    wavelength_step = (wavelengths[wavelength_length-1]-wavelengths[0])/(wavelength_length-1)
    f = interpolate.interp1d(wavelengths, spectrum)
    
    xnew = np.arange(wavelengths[0], wavelengths[wavelength_length-1], wavelength_step/10)  # waveltength interpolation
    
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    
    y_smooth = savgol_filter(ynew, 199, 3)  # smooth data to find the local maxima
    
    # plt.figure
    # plt.plot(wavelengths, spectrum, 'o', xnew, y_smooth, '-')   
    
    # plt.figure
    # plt.plot(xnew[10000:], y_smooth[10000:], '-') 
    
    peaks_ind, peaks_val = find_peaks(y_smooth, height=thershold_peak)#.1) #find the local maxima
    wavelength_max = xnew[peaks_ind]    # wavelength of the local maxima
    
    peaks_ind_lim = peaks_ind    
    peaks_ind_lim = [0] + peaks_ind[:]
    peaks_ind_lim = np.concatenate( ([0], peaks_ind, [len(xnew)-1]))
    
    lim_inf = [0 for ind in peaks_ind_lim[1:-1]]
    for ind in range(len(peaks_ind_lim)-2):
        lim_inf[ind] = peaks_ind_lim[ind+1] - round((peaks_ind_lim[ind+1]-peaks_ind_lim[ind])/2)
    lim_sup = [0 for ind in peaks_ind_lim[1:-1]]
    for ind in range(len(peaks_ind_lim)-2):
        lim_sup[ind] = peaks_ind_lim[ind+1] + round((peaks_ind_lim[ind+2]-peaks_ind_lim[ind+1])/2)  

    
    for inc in range(delete_peak):
        wavelength_max = np.delete(wavelength_max, 1) # delete doublet at 577 nm and 579 nm
        peaks_ind = np.delete(peaks_ind, 1) # delete doublet at 577 nm and 579 nm
        lim_inf = np.delete(lim_inf, 1) # delete doublet at 577 nm and 579 nm
        lim_sup = np.delete(lim_sup, 1) # delete doublet at 577 nm and 579 nm
    
      
    
    for ind in range(len(peaks_ind)):
        maxi = ynew[peaks_ind[ind]]#maxi = ynew[ind]
        if ind == 0:
            ref = maxi
            
        half_max = maxi/2
        
        left_value = abs(ynew[lim_inf[ind]:peaks_ind[ind]]-half_max)
        min_index = np.argmin(left_value) + lim_inf[ind]-1
        left_wavelength = xnew[min_index]
        
        right_value = abs(ynew[peaks_ind[ind]:lim_sup[ind]]-half_max)
        min_index = np.argmin(right_value) + peaks_ind[ind]-1
        right_wavelength = xnew[min_index]
        
        resolution = right_wavelength-left_wavelength
        print('peak ('+str(round(wavelength_max[ind]))+' nm) ==> resolution = '+str(round(resolution*100)/100)+' nm / Intensity = '+str(round(maxi*100)/100)+' counts / ratio = '+str(round(maxi/ref*100))+' %')



max_1 = np.amax(GT_1)
max_2 = np.amax(GT_2)
max_3 = np.amax(GT_3)

GT_2 = GT_2*max_1/max_2
GT_3 = GT_3*max_1/max_3

GT_tot = GT_1+GT_2+GT_3

plt.figure
plt.imshow(GT_tot)
#plt.colorbar();
#plt.grid()
plt.axis('off')

plt.savefig(reco_path + 'spectral_resolution_3_spot_dpi300.pdf', bbox_inches='tight',pad_inches = 0, dpi = 300)
plt.show()

plt.figure
plt.plot(wavelengths, spectrum_1/np.amax(spectrum_1))
plt.plot(wavelengths, spectrum_2/np.amax(spectrum_2), linestyle = 'dashed')
plt.plot(wavelengths, spectrum_3/np.amax(spectrum_3), linestyle = 'dotted')
plt.grid()
plt.xlabel('Wavelength $\lambda$ (in nm)')
plt.ylabel('Intensity $f$ (normalized)')
#plt.title('Spectral view in the spatial mean')
plt.legend(['1', '2', '3'])
plt.savefig(reco_path + 'spectral_resolution_3_spectra_dpi300.pdf', bbox_inches='tight',pad_inches = 0, dpi = 300)
plt.show

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

# cov_path = '../stats/new-nicolas/Cov_64x64.npy'
# mean_path = '../stats/new-nicolas/Average_64x64.npy'
# H_path = '../stats/new-nicolas/H.npy'
# model_root = '../models/new-nicolas/'

# model, device = setup_reconstruction(cov_path, mean_path, H_path, model_root, network_params)
# noise = load_noise('../noise-calibration/fit_model2.npz')

# reconstruction_params = {
#     'model': model,
#     'device': device,
#     'batches': 1,
#     'noise': noise,
# }

# F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(M.T, acquisition_parameters.wavelengths, 530, 730, 8, 0, noise)
# recon = reconstruct(model, device, F_bin[0:8192//4,:], 1, noise_bin)            
# plot_color(recon, wavelengths_bin)
# plt.savefig(nn_reco_path + '_reco_wavelength_binning.png')
# plt.show()

# #%% transfer data to girder
# transf.transfer_data_to_girder(metadata, acquisition_parameters, spectrometer_params, DMD_params, setup_version, data_folder_name, data_name)
# #%% delete plots
# Question = input("Do you want to delete the figures yes [y] ?  ")
# if Question == ("y") or Question == ("yes"):        
#     shutil.rmtree(overview_path)
#     print ("==> figures deleted")










