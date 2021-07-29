# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from singlepixel import *

def reconstruction_hadamard(acquisition_metadata, Q, M, N=64):
    ind_opt = acquisition_metadata.patterns[1::2]
    ind_opt = np.array(ind_opt)/2
    ind_opt = ind_opt - 1
    ind_opt = ind_opt.astype('int')
    M_breve = M[0::2,:] - M[1::2,:]
    M_Had = np.zeros((N*N, M.shape[1]))
    M_Had[ind_opt,:] = M_breve

    f = np.matmul(Q,M_Had) # Q.T = Q (Q transposed is Q)
    frames = np.reshape(f,(N,N,M.shape[1]))

    return frames

def reconstruction_raster(M, N=64):
    frames = np.reshape(M,(N,N,M.shape[1]))
    
    return frames

#%% Init
file = loadmat('../data/matlab.mat')
Q = file['Q']

spectrometer, DMD, DMD_initial_memory = init()
    
#%% Setup and acquire Hadamard
times = [1] # Integration time array

for integration_time in times:
    
    metadata = MetaData(
        output_directory='../data/17-03-2021-raster-starsector-light-off/',
        pattern_order_source='../communication/communication.txt',
        pattern_source='../Patterns/PosNeg/DMD_Hadamard_64x64/Hadamard_64x64_',
        experiment_name='/hadamard_' + str(integration_time) + '_OD13_025ms',
        light_source='White lamp LED',
        object='Starsector',
        filter='None',
        description=f'Hadamard acquisition with integration_time = {integration_time} ms. OD = 1.3. Ambient light off')
        
    acquisition_parameters = AcquisitionParameters(
        pattern_compression=1.0,
        pattern_dimension_x=64,
        pattern_dimension_y=64)
        
    spectrometer_params, DMD_params, wavelenghts = setup(
        spectrometer=spectrometer, 
        DMD=DMD,
        DMD_initial_memory=DMD_initial_memory,
        metadata=metadata, 
        acquisition_params=acquisition_parameters,
        integration_time=integration_time,)
        
    spectral_data, timestamps, measurement_time = acquire(
        ava=spectrometer,
        DMD=DMD,
        metadata=metadata,
        spectrometer_params=spectrometer_params,
        DMD_params=DMD_params,
        acquisition_params=acquisition_parameters,
        wavelengths=wavelenghts,)

    frames = reconstruction_hadamard(acquisition_parameters, Q, spectral_data)
    plt.figure()
    plt.imshow(np.sum(frames,axis=2))
    plt.title(f'Sum of all wavelenghts (Hadamard {integration_time} ms)')
    plt.show()
    plt.savefig(f'{metadata.output_directory[:-1]}{metadata.experiment_name}_recon.png')
    
    plt.figure()
    plt.plot(wavelenghts, spectral_data[0,:])
    plt.plot(wavelenghts, spectral_data[2,:])
    plt.legend(['1st pattern','3rd pattern'])
    plt.xlabel('Wavelenghts')
    plt.ylabel('Counts')
    plt.show()
    plt.savefig(f'{metadata.output_directory[:-1]}{metadata.experiment_name}_spect.png')

#%% Setup and acquire Raster
times = [64] # Integration time array

for integration_time in times:
    
    metadata = MetaData(
        output_directory='../data/17-03-2021-raster-starsector-light-off/',
        pattern_order_source='../communication/raster.txt',
        pattern_source='../Patterns/RasterScan_64x64/RasterScan_64x64_1_',
        experiment_name='/raster_' + str(integration_time) + '_OD13_64ms',
        light_source='White lamp LED',
        object='Starsector',
        filter='None',
        description=f'Raster acquisition with integration_time = {integration_time} ms.OD = 1.3. Ambient light off')
        
    acquisition_parameters = AcquisitionParameters(
        pattern_compression=1.0,
        pattern_dimension_x=64,
        pattern_dimension_y=64)
        
    spectrometer_params, DMD_params, wavelenghts = setup(
        spectrometer=spectrometer, 
        DMD=DMD,
        DMD_initial_memory=DMD_initial_memory,
        metadata=metadata, 
        acquisition_params=acquisition_parameters,
        integration_time=integration_time,)
        
    spectral_data, timestamps, measurement_time = acquire(
        ava=spectrometer,
        DMD=DMD,
        metadata=metadata,
        spectrometer_params=spectrometer_params,
        DMD_params=DMD_params,
        acquisition_params=acquisition_parameters,
        wavelengths=wavelenghts,)

    frames = reconstruction_raster(spectral_data)
    plt.figure()
    plt.imshow(np.sum(frames,axis=2))
    plt.title(f'Sum of all wavelenghts (Raster {integration_time} ms)')
    plt.show()
   # plt.savefig(f'{metadata.output_directory[:-1]}{metadata.experiment_name}_recon.png')
    
    plt.figure()
    plt.plot(wavelenghts, spectral_data[0,:])
    plt.plot(wavelenghts, spectral_data[2,:])
    plt.legend(['1st pattern','3rd pattern'])
    plt.xlabel('Wavelenghts')
    plt.ylabel('Counts')
    plt.show()
   # plt.savefig(f'{metadata.output_directory[:-1]}{metadata.experiment_name}_spect.png')

#%% Disconnect
disconnect(spectrometer, DMD)