# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:49:14 2023

@author: ducros
"""

#%%
from spas import ReconstructionParameters, setup_reconstruction

network_param = ReconstructionParameters(
    # Reconstruction network    
    M = 64*64,          # Number of measurements
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
        
cov_path = '../../stat/ILSVRC2012_v10102019/Cov_8_128x128.npy'
model_folder = '../../model_v2/'
      
model, device = setup_reconstruction(cov_path, model_folder, network_param)

#%% Load data and meta data
import numpy as np

# data
data_path = '../../pilot-spihim/cat_linear/Cat_LinearColoredFilter'
full_path =  data_path + '_spectraldata.npz'
meas = np.load(full_path)['spectral_data']

#%% Spectral binning
from spas import read_metadata, spectral_binning

# load meta data to get wavelength
meta_path = data_path + '_metadata.json'
_, acquisition_param, _, _ = read_metadata(meta_path)
wavelengths = acquisition_param.wavelengths 

# bin raw data
meas_bin, wavelengths_bin, _ = spectral_binning(meas.T, wavelengths, 530, 730, 4)

#%% Reorder and subsample
from spas.reconstruction_nn import reorder_subsample
meas_bin_2 = reorder_subsample(meas_bin, acquisition_param, network_param) 

#%% Reconstruct
from spas import reconstruct
rec = reconstruct(model, device, meas_bin_2, 1)

#%% Plot
from spas import plot_color 
plot_color(rec, wavelengths_bin)