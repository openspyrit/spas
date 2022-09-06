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

# Matlab patterns
#file = loadmat('../data/matlab.mat')
#Q = file['Q']

# fht patterns
#Q = Hadamard_Transform_Matrix(64)

#%% Reconstruction without NN
zoom = 1
########################## to be change ############################
setup_version = 'setup_v1.2'
data_folder_name = '2021-12-07_tomato_slice'#'2021-07-26-spectral-analysis'
data_name = 'tomato_slice_3_zoom_x', str(zoom) #'colored-siemens'
data_name = "".join(data_name)
data_name2 = 'whiteLamp_zoom_x', str(zoom)
data_name2 = "".join(data_name2)
########################### path ###################################
data_path = '../data/' + data_folder_name + '/' + data_name + '/' + data_name
data_path2 = '../data/' + data_folder_name + '/' + data_name2 + '/' + data_name2
had_reco_path = data_path + '_had_reco.npz'
had_reco_path2 = data_path2 + '_had_reco.npz'


########################## read raw data ###########################
file = np.load(data_path+'_spectraldata.npz')
M = file['spectral_data']#['arr_0']#['spectral_data']
Q = wh.walsh2_matrix(64)
file2 = np.load(data_path2+'_spectraldata.npz')
M2 = file2['spectral_data']#['arr_0']#['spectral_data']


metadata_path = data_path + '_metadata.json'
metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)

GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, M)
GT2 = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, M2)

GTn = GT/GT2/2
GTn[abs(GTn)>1]=0

metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
wavelengths = acquisition_metadata.wavelengths
F_bin, wavelengths_bin, bin_width = spectral_binning(GTn.T, acquisition_parameters.wavelengths, 530, 730, 8)

plt.figure(1)
plot_color(F_bin, wavelengths_bin)
plt.show()
plt.title('Spectral binning')

plt.figure(2)
plt.scatter(wavelengths, np.mean(np.mean(GTn,axis=1),axis=0))
plt.grid()
plt.xlabel('Lambda (nm)')
plt.title('Spectral view in the spatial mean')
plt.show()


if zoom == 12:
    sp1 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[50:53,2:6,20:2000], axis = 1), axis = 0), 20)
    sp2 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[34:37,31:36,20:2000], axis = 1), axis = 0), 20)
    sp3 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[19:25,52:57,20:2000], axis = 1), axis = 0), 20)

if zoom == 6:
    sp1 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[56:63,18:30,20:2000], axis = 1), axis = 0), 20)
    sp2 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[20:25,41:46,20:2000], axis = 1), axis = 0), 20)
    sp3 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[19:27,52:61,20:2000], axis = 1), axis = 0), 20)
    #sp4 =  scipy.ndimage.median_filter(GTn[8,7,:], 20)

if zoom == 1:
    sp1 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[20:27,4:21,20:2000], axis = 1), axis = 0), 10)
    sp2 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[9:17,36:42,20:2000], axis = 1), axis = 0), 10)
    sp3 =  scipy.ndimage.median_filter(np.mean(np.mean(GTn[34:54,7:35,20:2000], axis = 1), axis = 0), 10)

norm = 0
if norm :
    sp1 = sp1/np.amax(sp1)
    sp2 = sp2/np.amax(sp2)
    sp3 = sp3/np.amax(sp3)
    sp4 = sp4/np.amax(sp4)
    
# plt.figure(3)
fig, ax = plt.subplots()
ax.plot(wavelengths[20:2000], sp1.T, '-b', label = 'corner')
ax.plot(wavelengths[20:2000], sp2.T,'--r',  label = 'center')
ax.plot(wavelengths[20:2000], sp3.T,'.g',  label = 'bottom')
#ax.plot(wavelengths, sp4.T,'-c',  label = 'one point')
plt.grid()
plt.xlabel('Lambda (nm)')
plt.title('Spectral view')
leg = ax.legend();
plt.show()



fig, ax = plt.subplots()
ax.plot(wavelengths[20:2000], ((sp1+.1)/(sp2+.1)).T, '-b', label = 'corner/center')
ax.plot(wavelengths[20:2000], ((sp3+.1)/(sp2+.1)).T,'.g',  label = 'bottom/center')
plt.grid()
plt.xlabel('Lambda (nm)')
plt.title('Spectral view')
leg = ax.legend();
plt.show()

#%% Reconstruct with NN
network_params = ReconstructionParameters(
    img_size=64,
    CR=1024,
    denoise=True,
    epochs=40,
    learning_rate=1e-3,
    step_size=20,
    gamma=0.2,
    batch_size=256,
    regularization=1e-7,
    N0=50.0,
    sig=0.0,
    arch_name='c0mp',)

cov_path = '../stats/new-nicolas/Cov_64x64.npy'
mean_path = '../stats/new-nicolas/Average_64x64.npy'
H_path = '../stats/new-nicolas/H.npy'
model_root = '../models/new-nicolas/'

model, device = setup_reconstruction(cov_path, mean_path, H_path, model_root, network_params)
noise = load_noise('../noise-calibration/fit_model2.npz')

reconstruction_params = {
    'model': model,
    'device': device,
    'batches': 1,
    'noise': noise,
}

F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(M.T, acquisition_parameters.wavelengths, 530, 730, 8, 0, noise)
recon = reconstruct(model, device, F_bin[0:8192//4,:], 1, noise_bin)            
plot_color(recon, wavelengths_bin)
plt.savefig(nn_reco_path + '_reco_wavelength_binning.png')
plt.show()

#%% transfer data to girder
transf.transfer_data_to_girder(metadata, acquisition_parameters, spectrometer_params, DMD_params, setup_version, data_folder_name, data_name)
#%% delete plots
Question = input("Do you want to delete the figures yes [y] ?  ")
if Question == ("y") or Question == ("yes"):        
    shutil.rmtree(overview_path)
    print ("==> figures deleted")










