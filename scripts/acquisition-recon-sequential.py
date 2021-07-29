"""
Example of an acquisition followed by a reconstruction using 100 % of the
Hadamard patterns and then, a reconstruction using 1/4 of the patterns 
(subsampled) with a DenoiCompNet model and a noise model. Reconstructions are 
performed after the acquisition and not in "real-time".
"""

from spas import *

import numpy as np
import spyrit.misc.walsh_hadamard as wh
    
#%% Init
spectrometer, DMD, DMD_initial_memory = init()
    
#%% Setup and acquire
metadata = MetaData(
    output_directory='../data/2021-07-21-walsh-patterns',
    pattern_order_source='../stats/new-nicolas/Cov_64x64.npy',
    pattern_source='../Patterns/PosNeg/DMD_Walsh_64x64',
    pattern_prefix='Walsh_64x64',
    experiment_name='walsh-siemens-star1',
    light_source='White lamp LED',
    object='Siemens star',
    filter='Diffuser',
    description='Testing acquisition using torch-generated Walsh patterns')
    
acquisition_parameters = AcquisitionParameters(
    pattern_compression=1.0,
    pattern_dimension_x=64,
    pattern_dimension_y=64)
    
spectrometer_params, DMD_params = setup(
    spectrometer=spectrometer, 
    DMD=DMD,
    DMD_initial_memory=DMD_initial_memory,
    metadata=metadata, 
    acquisition_params=acquisition_parameters,
    integration_time=1.0)

    
#%% Setup reconstruction

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

#%% Acquire
spectral_data = acquire(
    ava=spectrometer,
    DMD=DMD,
    metadata=metadata,
    spectrometer_params=spectrometer_params,
    DMD_params=DMD_params,
    acquisition_params=acquisition_parameters,
    repetitions=1,
    reconstruct=False)

#%% Reconstruction without NN

#file = loadmat('./data/matlab.mat')
#Q = file["Q"]
Q = wh.walsh2_matrix(64)

GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, spectral_data)

F_bin, wavelengths_bin, bin_width = spectral_binning(GT.T, acquisition_parameters.wavelengths, 530, 730, 8)

plot_color(F_bin, wavelengths_bin)

#%% Reconstruct with NN

F_bin, wavelengths_bin, bin_width, noise_bin = spectral_binning(spectral_data.T, acquisition_parameters.wavelengths, 530, 730, 8, noise)
recon = reconstruct(model, device, F_bin[0:8192//4,:], 1, noise_bin)           
plot_color(recon, wavelengths_bin)

#%% Disconnect
disconnect(spectrometer, DMD)
